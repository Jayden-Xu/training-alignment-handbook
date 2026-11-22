import contextvars
import functools
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ["CASE2_AUDIT_DIR"])
OLD_STEP = contextvars.ContextVar("old_step")
AUDIT_STEP = "case2_audit_step"


def write(name, data):
    path = ROOT / name
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, sort_keys=True) + "\n")
    temp.replace(path)


def stats(rollout, candidate, mask):
    import numpy as np
    left = rollout.detach().float()[mask].cpu().numpy()
    right = candidate.detach().float()[mask].cpu().numpy()
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite valid response logprob in audit")
    delta = right.astype(np.float64) - left.astype(np.float64)
    different_bits = left.view(np.uint32) != right.view(np.uint32)
    signed_zero = different_bits & (left == 0) & (right == 0)
    return {
        "tokens": int(delta.size),
        "mismatches": int(np.count_nonzero(different_bits)),
        "value_mismatches": int(np.count_nonzero(left != right)),
        "signed_zero_mismatches": int(np.count_nonzero(signed_zero)),
        "k3_sum": float(np.sum(np.expm1(delta) - delta)),
        "max_abs": float(np.max(np.abs(delta), initial=0)),
    }


def trainer(module):
    cls = module.PPOTrainer
    compute_old = cls._compute_old_log_prob
    debug = module.calculate_debug_metrics
    update_actor = cls._update_actor

    @functools.wraps(compute_old)
    def old(self, batch, metrics):
        token = OLD_STEP.set(self.global_steps)
        try:
            return compute_old(self, batch, metrics)
        finally:
            OLD_STEP.reset(token)

    @functools.wraps(debug)
    def compare(data):
        metrics = debug(data)
        batch = data.batch
        observed = stats(batch["rollout_log_probs"], batch["old_log_probs"], batch["response_mask"].bool())
        observed["native_mismatches"] = int(metrics["training/rollout_logprobs_mismatch_count"])
        write(f"old-step-{OLD_STEP.get()}.json", observed)
        return metrics

    @functools.wraps(update_actor)
    def actor(self, batch, metrics):
        batch.extra_info[AUDIT_STEP] = self.global_steps
        return update_actor(self, batch, metrics)

    cls._compute_old_log_prob = old
    module.calculate_debug_metrics = compare
    cls._update_actor = actor


def loss(module):
    original = module.ppo_loss
    @functools.wraps(original)
    def observe(config, model_output, data, dp_group=None):
        import torch
        step = int(data.get_non_tensor(AUDIT_STEP))
        rank = torch.distributed.get_rank()
        with torch.no_grad():
            logp = module.no_padding_2_padding(model_output["log_probs"], data)
            raw = data.select("rollout_log_probs", "response_mask").to_padded_tensor()
            row = stats(raw["rollout_log_probs"], logp, raw["response_mask"].bool())
        with (ROOT / f"actor-step-{step}-rank-{rank}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return original(config, model_output, data, dp_group)
    module.ppo_loss = observe


def weight_hash(tensor):
    import torch
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return [str(tensor.dtype), list(tensor.shape), hashlib.sha256(memoryview(value.numpy())).hexdigest()]


def transport(module):
    sender, receiver = module.BucketedWeightSender, module.BucketedWeightReceiver
    launch_send, launch_receive = sender.async_send_weights, receiver.receive_weights
    send_round = recv_round = 0

    @functools.wraps(launch_send)
    async def send(self, weights):
        import torch
        from verl.workers.rollout.utils import ensure_async_iterator
        if torch.distributed.get_rank() != 0:
            return await launch_send(self, weights)
        nonlocal send_round
        round_id = send_round
        send_round += 1
        hashes = {}
        async def record():
            async for name, tensor in ensure_async_iterator(weights):
                hashes[name] = weight_hash(tensor)
                yield name, tensor
        result = await launch_send(self, record())
        write(f"sync-{round_id}-sender.json", hashes)
        return result

    @functools.wraps(launch_receive)
    def receive(self, on_bucket_received):
        nonlocal recv_round
        replica = int(os.environ["VERL_REPLICA_RANK"])
        round_id = recv_round
        recv_round += 1
        hashes = {}
        def record(weights, is_last):
            for name, tensor in weights:
                hashes[name] = weight_hash(tensor)
            return on_bucket_received(weights, is_last)
        result = launch_receive(self, record)
        write(f"sync-{round_id}-receiver-{replica}.json", hashes)
        return result

    sender.async_send_weights = send
    receiver.receive_weights = receive


def rollout(module):
    cls = module.vLLMColocateWorkerExtension
    original = cls.update_weights_from_ipc
    round_id = 0

    @functools.wraps(original)
    def after(self, *args, **kwargs):
        nonlocal round_id
        result = original(self, *args, **kwargs)
        replica = int(os.environ["VERL_REPLICA_RANK"])
        hashes = {name: weight_hash(tensor) for name, tensor in self.model_runner.model.named_parameters()}
        write(f"sync-{round_id}-loaded-{replica}.json", hashes)
        round_id += 1
        return result

    cls.update_weights_from_ipc = after


def check_sync(step):
    def read(name):
        return json.loads((ROOT / f"sync-{step}-{name}.json").read_text())
    sender = read("sender")
    loaded = read("loaded-0")
    if not sender or not loaded:
        raise ValueError(f"Empty weight digest coverage at sync {step}")
    for replica in range(16):
        received = read(f"receiver-{replica}")
        physical = read(f"loaded-{replica}")
        if sender != received:
            bad = [name for name in sender.keys() | received.keys() if sender.get(name) != received.get(name)]
            raise ValueError(f"Sync {step}, replica {replica}: IPC SHA256 mismatch in {bad[:5]}")
        if loaded != physical:
            bad = [name for name in loaded.keys() | physical.keys() if loaded.get(name) != physical.get(name)]
            raise ValueError(f"Sync {step}, replica {replica}: post-load SHA256 mismatch in {bad[:5]}")
    result = {
        "step": step, "replicas": 16,
        "payload_tensors_per_replica": len(sender), "post_load_parameters_per_replica": len(loaded),
        "payload_mismatches": 0, "post_load_replica_mismatches": 0,
    }
    write(f"sync-{step}-verified.json", result)
    print("Gate 0 SHA-256 verified", result, flush=True)


def sync(module):
    cls = module.PPOTrainerSync
    def wrap(method):
        @functools.wraps(method)
        def check(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            check_sync(self.global_steps)
            return result
        return check
    for name in ("on_init_end", "on_step_end"):
        setattr(cls, name, wrap(getattr(cls, name)))
