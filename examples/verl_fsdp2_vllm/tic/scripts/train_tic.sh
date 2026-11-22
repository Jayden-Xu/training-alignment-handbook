#!/usr/bin/env bash
# train_tic.sh <baseline|aligned> <rollout TP> [extra Hydra overrides]
# Starts verl.trainer.main_ppo for one arm. Normally called by train_tic_job.sh, which
# c9_suite.sh starts (see c9_suite.sh for the whole chain).
#
# Layout: CASE2_TIC_ROOT is the tic/ folder (the one that holds scripts/); the patched verl
# checkout goes at $CASE2_TIC_ROOT/verl, and $case_root/audit is the top-level audit/ folder.
#
# A. Provided by the env file ${CASE2_ENV_SH}, which you prepare:
#    case_root        case folder (holds audit/)
#    dataset          folder with train.parquet and test.parquet
#    py               the environment's python
#    PATH, LD_LIBRARY_PATH  for the environment: the CUDA 13 NVIDIA libraries from
#                     site-packages, and the libstdc++ that the locally built Transformer Engine needs
#    PYTHONDONTWRITEBYTECODE=1
#    cache folders    HF_HOME, VLLM_CACHE_ROOT, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR,
#                     FLASHINFER_WORKSPACE_BASE, XDG_CACHE_HOME, RAY_TMPDIR
#    HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
#    The env file must unset RAY_ADDRESS VERL_IPC_NAMESPACE VERL_RAY_JOB_ID
#    VERL_FULL_DETERMINISM VLLM_BATCH_INVARIANT, and cd to $case_root.
#
# B. Set at launch (values used in the formal runs):
#    CASE2_TIC_ROOT, CASE2_C9_SUITE, CASE2_C9_ALIGNED_SUITE (baseline-only runs),
#    CASE2_BATCH_SIZE=16, CASE2_GPUS=16, CASE2_ROOT_MODEL (model path), CASE2_TRAIN_STEPS=10,
#    CASE2_ROLLOUT_MEMORY=0.95, CASE2_OBSERVE_SHA=1, CASE2_FINAL_SCORE_STEP=0, CASE2_C5_REPLAY=0,
#    VLLM_USE_V2_MODEL_RUNNER=1, CUDA_VISIBLE_DEVICES=0,1,...,15
#
# C. Set by c9_suite.sh: CASE2_JOB, CASE2_ARM, CASE2_C9_ARM, CASE2_C9_MASTER_PORT, CASE2_PHASE
#    and the suite's run folder; train_tic_job.sh then sets the arm's run folder
#    (CASE2_RUN_DIR, CASE2_TIC_RUN) and CASE2_STATUS.
#
# D. Set here: VLLM_BATCH_INVARIANT=1 and the compilation config from
#    $CASE2_TIC_ROOT/steps/C3_graph_only_config.json, in the aligned arm only;
#    PYTHONPATH=$CASE2_TIC_ROOT/verl:$CASE2_TIC_ROOT/scripts/audit:$case_root/audit.
set -euo pipefail
source "${CASE2_ENV_SH:?set CASE2_ENV_SH to your env file}"
mode="${1:?baseline or aligned}"
tp="${2:?TP confirmed by standalone smoke}"
shift 2
case "$mode" in
  baseline) invariant=false; compilation_config='{"mode":0,"cudagraph_mode":"FULL"}' ;;
  aligned)
    invariant=true
    export VLLM_BATCH_INVARIANT=1
    compilation_config="$(tr -d '\n' <"$CASE2_TIC_ROOT/steps/C3_graph_only_config.json")"
    ;;
  *) exit 2 ;;
esac
unset CASE2_AUDIT_DIR
export PYTHONPATH="$CASE2_TIC_ROOT/verl:$CASE2_TIC_ROOT/scripts/audit:$case_root/audit"
printf 'Case 2 TIC mode=%s verl=%s TP=%s\n' "$mode" "$(git -C "$CASE2_TIC_ROOT/verl" rev-parse HEAD)" "$tp"
rollout_mem="${CASE2_ROLLOUT_MEMORY:-0.95}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
: "${CUDA_VISIBLE_DEVICES:?set in tmux environment}" "${CASE2_GPUS:?}" "${CASE2_ROOT_MODEL:?}" "${CASE2_STATUS:?}" "${CASE2_RUN_DIR:?}"
exec "$py" -Bu -m verl.trainer.main_ppo \
  "data.train_files=$dataset/train.parquet" "data.val_files=$dataset/test.parquet" \
  "data.train_batch_size=${CASE2_BATCH_SIZE:-16}" data.max_prompt_length=512 data.max_response_length=512 \
  data.filter_overlong_prompts=true data.truncation=error data.seed=42 \
  +data.apply_chat_template_kwargs.reasoning_effort=low \
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=false \
  algorithm.use_kl_in_reward=false algorithm.kl_ctrl.kl_coef=0.0 \
  "actor_rollout_ref.model.path=$CASE2_ROOT_MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.use_fused_kernels=false actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  "actor_rollout_ref.actor.fsdp_config.batch_invariant=$invariant" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=128 \
  actor_rollout_ref.actor.optim.lr=1e-6 "actor_rollout_ref.actor.ppo_mini_batch_size=${CASE2_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.loss_agg_mode=token-mean \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config='$compilation_config'" \
  actor_rollout_ref.rollout.name=vllm "actor_rollout_ref.rollout.tensor_model_parallel_size=$tp" \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs=true actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_mem" actor_rollout_ref.rollout.max_model_len=1024 \
  "trainer.n_gpus_per_node=$CASE2_GPUS" trainer.nnodes=1 "trainer.total_training_steps=$CASE2_TRAIN_STEPS" \
  trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.resume_mode=disable 'trainer.logger=["console"]' \
  trainer.project_name=case2-tic "trainer.experiment_name=$mode" \
  "trainer.default_local_dir=$CASE2_RUN_DIR/checkpoints" \
  '+ray_kwargs.ray_init.address=local' +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_TIC_RUN=$CASE2_RUN_DIR" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_STATUS=$CASE2_STATUS" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_C5_REPLAY='${CASE2_C5_REPLAY:-0}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_OBSERVE_SHA='${CASE2_OBSERVE_SHA:-0}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_FINAL_SCORE_STEP='${CASE2_FINAL_SCORE_STEP:-0}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_C9_ARM='${CASE2_C9_ARM:-}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CASE2_C9_MASTER_PORT='${CASE2_C9_MASTER_PORT:-}'" \
  "hydra.run.dir=$CASE2_RUN_DIR/hydra" "$@"
