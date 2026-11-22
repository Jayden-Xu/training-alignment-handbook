"""Store raw valid-token observations at the actual PPO old-policy boundary."""
import contextvars
import functools
import json
import math
import os
from pathlib import Path

import torch

RUN = Path(os.environ["CASE2_TIC_RUN"])
STEP = contextvars.ContextVar("case2_tic_step")
REPLAY = contextvars.ContextVar("case2_tic_replay", default=None)
REPLAY_KEY = "case2_old_replay"
PACK_KEY = "micro_batch_size_per_gpu"
C9_ARM = os.environ.get("CASE2_C9_ARM", "")


def observe_ppo_runtime_env(module):
    original = module.get_ppo_ray_runtime_env

    @functools.wraps(original)
    def baseline_env(*args, **kwargs):
        runtime = original(*args, **kwargs)
        env = runtime["env_vars"]
        if os.environ.get("VLLM_BATCH_INVARIANT") is not None or env.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
            raise RuntimeError("C9 baseline driver must leave VLLM_BATCH_INVARIANT unset")
        # verl supplies "0" for an absent key. Preserve the experiment's
        # literal unset contract for all Ray children.
        env.pop("VLLM_BATCH_INVARIANT", None)
        return runtime

    module.get_ppo_ray_runtime_env = baseline_env


def prove_training_path(module):
    cls = module.FSDPEngine
    build = cls._build_module

    @functools.wraps(build)
    def traced(self):
        model = build(self)
        backbone = model.model
        layers = backbone.layers

        def origin(method):
            return f"{method.__module__}.{method.__qualname__}"

        row = {
            "rank": self.rank, "arm": C9_ARM, "layers": len(layers),
            "master_port": os.environ.get("MASTER_PORT"),
            "batch_invariant": bool(self.engine_config.batch_invariant),
            "gpt_oss_case2": bool(self._gpt_oss_case2),
            "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
            "backbone": origin(backbone.forward),
            "decoder": [origin(layer.forward) for layer in layers],
            "attention": origin(layers[0].self_attn.forward),
            "attention_backend": model.config._attn_implementation,
            "norm": origin(layers[0].input_layernorm.forward),
            "lm_head": origin(model.lm_head.forward),
        }
        (RUN / f"train-path-{self.rank}.json").write_text(json.dumps(row, indent=2) + "\n")
        if C9_ARM == "baseline":
            correct = (not row["gpt_oss_case2"] and not row["batch_invariant"]
                       and row["vllm_batch_invariant"] is None)
        else:
            bound = [row["backbone"], *row["decoder"], row["lm_head"]]
            marker = "verl.models.transformers.gpt_oss_case2."
            correct = (row["gpt_oss_case2"] and row["batch_invariant"]
                       and all(name.startswith(marker) for name in bound))
        if not correct:
            raise RuntimeError(f"C9 unexpected training implementation on rank {self.rank}. See train-path-{self.rank}.json")
        if self.rank == 0:
            print(f"C9 training path arm={C9_ARM} gpt_oss_case2={row['gpt_oss_case2']} "
                  f"batch_invariant={row['batch_invariant']} vllm_batch_invariant={row['vllm_batch_invariant']!r} "
                  f"backbone={row['backbone']} decoder={row['decoder'][0]} "
                  f"attention={row['attention']} backend={row['attention_backend']} "
                  f"norm={row['norm']}", flush=True)
        return model

    cls._build_module = traced


def old_compare(a, b):
    mask = a["mask"]
    if not torch.equal(mask, b["mask"]) or not torch.equal(a["responses"], b["responses"]):
        raise RuntimeError("Old replay changed logical response IDs or response mask")
    x, y = a["old"][mask], b["old"][mask]
    delta = y.double() - x.double()
    return {"mismatch": int((x.view(torch.int32) != y.view(torch.int32)).sum()),
            "tokens": len(x), "max_abs": delta.abs().max().item()}


def observe_packing(module):
    if C9_ARM:
        prove_training_path(module)
    if os.environ.get("CASE2_C5_REPLAY") != "1":
        return
    cls = module.FSDPEngineWithLMHead
    prepare = cls.prepare_model_inputs

    @functools.wraps(prepare)
    def witnessed(self, micro_batch):
        result = prepare(self, micro_batch)
        if REPLAY_KEY in micro_batch.keys():
            label = micro_batch[REPLAY_KEY]
            lengths = micro_batch["input_ids"].offsets().diff().tolist()
            rank = torch.distributed.get_rank()
            with (RUN / f"packing-{label}-{rank}.jsonl").open("a") as output:
                output.write(json.dumps(lengths) + "\n")
        return result
    cls.prepare_model_inputs = witnessed


def observe_trainer(module):
    cls = module.PPOTrainer
    compute_old = cls._compute_old_log_prob
    debug = module.calculate_debug_metrics
    update = cls._update_actor
    compute_metrics = cls._compute_metrics
    if C9_ARM:
        port = int(os.environ["CASE2_C9_MASTER_PORT"])
        # verl assigns MASTER_PORT itself. Ask its existing Ray allocator to
        # recheck the launcher-selected port on the rendezvous node.
        module.RayWorkerGroup = functools.partial(module.RayWorkerGroup, master_port_range=[port, port + 1])
    c8 = os.environ.get("CASE2_FINAL_SCORE_STEP") == "3"

    @functools.wraps(compute_old)
    def old_logprob(self, batch, metrics):
        mark = STEP.set(self.global_steps)
        try:
            if os.environ.get("CASE2_C5_REPLAY") != "1":
                batch = compute_old(self, batch, metrics)
                if C9_ARM == "aligned":
                    step = self.global_steps
                    stem = f"old-step-{step}"
                    row = json.loads((RUN / f"{stem}.json").read_text())
                    if row["mismatch"] > 0 or row["k3_kl"] != 0:
                        frozen_path = RUN / f"frozen-step-{step}.pt"
                        fault = {**row, "frozen_batch": frozen_path.name, "scores": f"{stem}.pt"}
                        fault_path = RUN / "alignment-failure.json"
                        fault_path.write_text(json.dumps(fault, indent=2) + "\n")
                        frozen = module.tq.kv_batch_get(
                            keys=batch.keys, partition_id=batch.partition_id,
                            select_fields=["input_ids", "position_ids", "response_mask",
                                           "responses", "rollout_log_probs", "old_log_probs"],
                        )
                        temporary = frozen_path.with_suffix(".tmp")
                        torch.save({"step": step, "keys": [str(key) for key in batch.keys],
                                    **{name: frozen[name].cpu() for name in frozen.keys()}}, temporary)
                        temporary.replace(frozen_path)
                        raise RuntimeError(f"C9 aligned stop before step {step} update: {row}")
                return batch
            snapshots = {}
            evidence = {}
            original_keys = list(batch.keys)
            frozen = module.tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id,
                select_fields=["input_ids", "position_ids", "response_mask", "responses", "rollout_log_probs"],
            )
            torch.save({"keys": [str(key) for key in original_keys],
                        **{name: frozen[name].cpu() for name in frozen.keys()}}, RUN / "frozen-c5.pt")
            for label, packing in (("A1", 1), ("A2", 1), ("C", 2)):
                replay = REPLAY.set(label)
                batch.extra_info.update({REPLAY_KEY: label, PACK_KEY: packing})
                try:
                    batch = compute_old(self, batch, metrics)
                finally:
                    REPLAY.reset(replay)
                if original_keys != list(batch.keys):
                    raise RuntimeError(f"TransferQueue changed request order during {label}")
                stem = f"old-step-{self.global_steps}-{label}"
                snapshots[label] = torch.load(RUN / f"{stem}.pt", weights_only=True)
                evidence[label] = json.loads((RUN / f"{stem}.json").read_text())
                traces = sorted(RUN.glob(f"packing-{label}-*.jsonl"))
                if len(traces) != self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes:
                    raise RuntimeError(f"Incomplete rank packing witnesses for {label}")
                for trace in traces:
                    lengths = [json.loads(line) for line in trace.read_text().splitlines()]
                    if not lengths or any(len(micro_batch) != packing for micro_batch in lengths):
                        raise RuntimeError(f"Native packing mismatch {label} in {trace.name}")
            batch.extra_info.pop(REPLAY_KEY, None)
            batch.extra_info.pop(PACK_KEY, None)
            invariance = {"A/A": old_compare(snapshots["A1"], snapshots["A2"]),
                          "A/C": old_compare(snapshots["A1"], snapshots["C"])}
            (RUN / "native-c5.json").write_text(json.dumps({
                "rollout_pairs": evidence, "training_pairs": invariance,
            }, indent=2) + "\n")
            if any(row["mismatch"] for row in (*evidence.values(), *invariance.values())):
                raise RuntimeError(f"Native C5 replay failed: {evidence=} {invariance=}")
            return batch
        finally:
            STEP.reset(mark)

    @functools.wraps(debug)
    def compare(data):
        metrics = debug(data)
        step = STEP.get(None)
        if step is None:
            return metrics
        batch = data.batch
        mask = batch["response_mask"].bool().cpu()
        rollout = batch["rollout_log_probs"].detach().cpu()
        old = batch["old_log_probs"].detach().cpu()
        valid_rollout, valid_old = rollout[mask], old[mask]
        if rollout.dtype != torch.float32 or old.dtype != torch.float32 or not mask.any():
            raise RuntimeError("Expected nonempty FP32 old and rollout logprobs")
        if not torch.isfinite(valid_rollout).all() or not torch.isfinite(valid_old).all():
            raise RuntimeError("Non-finite valid logprobs")
        delta = valid_old.double() - valid_rollout.double()
        row = {"step": step, "mismatch": (valid_rollout.view(torch.int32) != valid_old.view(torch.int32)).sum().item(),
               "tokens": delta.numel(), "k3_kl": (delta.expm1() - delta).mean().item(),
               "max_abs": delta.abs().max().item()}
        label = REPLAY.get()
        stem = f"old-step-{step}" + (f"-{label}" if label else "")
        torch.save({"rollout": rollout, "old": old, "mask": mask,
                    "responses": batch["responses"].detach().cpu()}, RUN / f"{stem}.pt")
        (RUN / f"{stem}.json").write_text(json.dumps(row) + "\n")
        status = Path(os.environ["CASE2_STATUS"])
        tmp = status.with_suffix(".tmp")
        tmp.write_text(f"phase=native verl old logprob\nstep={step} {label or ''}\n"
                       f"mismatch={row['mismatch']}/{row['tokens']}\nKL={row['k3_kl']:.9g}\n")
        tmp.replace(status)
        if c8 and row["mismatch"]:
            raise RuntimeError(f"C8 old score mismatch: {row}")
        return metrics

    @functools.wraps(update)
    def update_actor(self, batch, metrics):
        if self.global_steps == int(os.environ.get("CASE2_FINAL_SCORE_STEP", "0")):
            return batch
        reward_stats = {}
        if c8:
            data = module.tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id,
                select_fields=["advantages", "response_mask", "rm_scores"],
            ).to_padded_tensor()
            mask = data["response_mask"].bool()
            adv = data["advantages"][mask].float()
            rewards = data["rm_scores"].float().sum(dim=-1)
            assert torch.isfinite(adv).all() and torch.isfinite(rewards).all()
            reward_stats = {"reward_min": rewards.min().item(), "reward_max": rewards.max().item(),
                            "reward_std": rewards.std(unbiased=False).item(),
                            "nonzero_advantages": adv.count_nonzero().item(), "valid_advantages": adv.numel()}
            if reward_stats["reward_std"] <= 0 or reward_stats["nonzero_advantages"] == 0:
                raise RuntimeError(f"C8 degenerate reward or GRPO advantage: {reward_stats}")
        if C9_ARM:
            data = module.tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id,
                select_fields=["advantages", "response_mask"],
            ).to_padded_tensor()
            mask = data["response_mask"].bool()
            advantages = data["advantages"]
            if not torch.isfinite(advantages[mask]).all():
                raise RuntimeError("C9 non-finite valid response advantage")
            reward_stats = {"nonzero_adv": ((advantages != 0) & mask).any(dim=-1).count_nonzero().item(),
                            "responses": mask.any(dim=-1).count_nonzero().item()}
        result = update(self, batch, metrics)
        grad_norm = metrics.get("actor/grad_norm")
        if c8 and (grad_norm is None or not math.isfinite(float(grad_norm)) or float(grad_norm) <= 0):
            raise RuntimeError(f"C8 expected positive finite grad norm, got {grad_norm}")
        with (RUN / "updates.jsonl").open("a") as output:
            output.write(json.dumps({"step": self.global_steps,
                                     "grad_norm": float(grad_norm) if grad_norm is not None else None,
                                     **reward_stats}) + "\n")
        return result

    @functools.wraps(compute_metrics)
    def timing(self, batch, metrics, timing_raw, global_steps, epoch):
        result = compute_metrics(self, batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
        with (RUN / "timings.jsonl").open("a") as output:
            output.write(json.dumps({"step": global_steps, "seconds": timing_raw}) + "\n")
        return result

    if c8 or C9_ARM:
        cls._compute_metrics = timing
    cls._compute_old_log_prob = old_logprob
    module.calculate_debug_metrics = compare
    cls._update_actor = update_actor


def observe_transport(module):
    if os.environ.get("CASE2_OBSERVE_SHA") == "1":
        os.environ["CASE2_AUDIT_DIR"] = str(RUN)
        import gate0_audit
        gate0_audit.transport(module)


def observe_rollout(module):
    if os.environ.get("CASE2_OBSERVE_SHA") == "1":
        os.environ["CASE2_AUDIT_DIR"] = str(RUN)
        import gate0_audit
        gate0_audit.rollout(module)
        cls = module.vLLMColocateWorkerExtension
        load = cls.update_weights_from_ipc

        @functools.wraps(load)
        def witness(self, *args, **kwargs):
            result = load(self, *args, **kwargs)
            replica = os.environ["VERL_REPLICA_RANK"]
            runner = self.model_runner
            cfg = runner.vllm_config
            rope = runner.model.model.layers[0].attn.rotary_emb.cos_sin_cache
            witness = {"runner": f"{type(runner).__module__}.{type(runner).__name__}",
                       "mode": str(cfg.compilation_config.mode),
                       "graph": str(cfg.compilation_config.cudagraph_mode),
                       "ops": list(cfg.compilation_config.custom_ops),
                       "aot": os.environ.get("VLLM_USE_AOT_COMPILE"),
                       "vllm_bi": os.environ.get("VLLM_BATCH_INVARIANT"),
                       "rope": gate0_audit.weight_hash(rope),
                       "replica": int(replica)}
            path = RUN / f"runtime-{replica}.json"
            if not path.exists():
                path.write_text(json.dumps(witness, indent=2) + "\n")
            return result

        cls.update_weights_from_ipc = witness
