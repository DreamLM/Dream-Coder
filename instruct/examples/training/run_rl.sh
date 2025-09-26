#!/bin/bash

set -x

# job info
CKPT=$1
if [ -z "$CKPT" ]; then
    echo "Usage: $0 <checkpoint_path> <sandbox_fusion_endpoint>"
    exit 1
fi

SANDBOX_FUSION_ENDPOINT=$2
if [ -z "$SANDBOX_FUSION_ENDPOINT" ]; then
    echo "Usage: $0 <checkpoint_path> <sandbox_fusion_endpoint>"
    exit 1
fi
export SANDBOX_FUSION_ENDPOINT=$SANDBOX_FUSION_ENDPOINT

JOBNAME="v3pluscodeio_reslen768_n16_lr5e-7_bs96_cartp0.1_ar_nolossonpad_rolloutbs1_ppoepoch2_cache_dynamic_from35k_coupled_cliphigh_topp0.95"
TRAIN_DATASET="Dream-org/Dream-Coder-RL-17k"
# get return path without extracting
TRAIN_ROOT=$(huggingface-cli download --repo-type dataset $TRAIN_DATASET)
TRAIN_FILES=$(find "$TRAIN_ROOT" -name "*.parquet" | awk '{printf "'%s',", $0}' | sed 's/,$//')

python3 -m src.trainer.main_ppo \
   +actor_rollout_ref.actor.clip_ratio_high=0.28 \
   actor_rollout_ref.rollout.top_p=0.95 \
   algorithm.adv_estimator=grpo \
   reward_model.reward_manager=ctrl \
   reward_model.save_metadata=True \
   data.train_files="[$TRAIN_FILES]" \
   data.val_files="[$TRAIN_FILES]" \
   data.train_batch_size=96 \
   data.max_prompt_length=512 \
   data.max_response_length=768 \
   data.filter_overlong_prompts=True \
   data.truncation='error' \
   actor_rollout_ref.model.path=${CKPT} \
   +actor_rollout_ref.model.trust_remote_code=True \
   actor_rollout_ref.actor.no_loss_on_pad=True \
   actor_rollout_ref.actor.optim.lr=5e-7 \
   actor_rollout_ref.actor.context_adaptive_reweight_p=0.1 \
   actor_rollout_ref.model.use_remove_padding=False \
   actor_rollout_ref.actor.ppo_mini_batch_size=96 \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
   +actor_rollout_ref.rollout.micro_batch_size=1 \
   actor_rollout_ref.actor.dynamic_filtering=True \
   actor_rollout_ref.actor.ppo_epochs=2 \
   actor_rollout_ref.actor.mask_epochs=2 \
   actor_rollout_ref.actor.t_schedule=couple \
   actor_rollout_ref.actor.use_kl_loss=False \
   actor_rollout_ref.actor.kl_loss_coef=0.001 \
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.entropy_coeff=0 \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.fsdp_config.param_offload=False \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
   actor_rollout_ref.rollout.accelerator.enable=True \
   actor_rollout_ref.rollout.accelerator.type="fast_dllm" \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
   actor_rollout_ref.rollout.name=hf \
   actor_rollout_ref.rollout.n=16 \
   actor_rollout_ref.rollout.temperature=1.0 \
   actor_rollout_ref.rollout.diffusion_steps=768 \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   algorithm.kl_ctrl.kl_coef=0.001 \
   +trainer.val_before_train=False \
   trainer.balance_batch=False \
   trainer.critic_warmup=0 \
   trainer.logger=['console','wandb'] \
   trainer.project_name="diff-rl" \
   trainer.experiment_name=$JOBNAME \
   trainer.n_gpus_per_node=8 \
   trainer.nnodes=1 \
   trainer.save_freq=10 \
   trainer.test_freq=-1 \
   trainer.total_epochs=15
