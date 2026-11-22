"""Observe Case 2's native old-logprob and weight transfer paths."""
import os
import sys
from importlib.machinery import PathFinder

TARGETS = {
    "verl.workers.engine.fsdp.transformer_impl": "observe_packing",
    "verl.trainer.constants_ppo": "observe_ppo_runtime_env",
    "verl.trainer.ppo.v1.trainer_base": "observe_trainer",
    "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": "observe_transport",
    "verl.workers.rollout.vllm_rollout.utils": "observe_rollout",
}


class AuditFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        if fullname == "verl.trainer.constants_ppo" and os.environ.get("CASE2_C9_ARM") != "baseline":
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None
        execute = spec.loader.exec_module

        def audited(module):
            execute(module)
            import tic_audit
            getattr(tic_audit, TARGETS[fullname])(module)

        spec.loader.exec_module = audited
        return spec


if os.environ.get("CASE2_TIC_RUN"):
    sys.meta_path.insert(0, AuditFinder())
