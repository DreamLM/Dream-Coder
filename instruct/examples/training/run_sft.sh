JOBNAME="lingcoder_bs576"
CKPT_PATH="Dream-org/Dream-Coder-v0-Base-7B"
REWEIGHT="cart"
LR=5e-6

torchrun --nnodes=1 --nproc_per_node=8 \
    -m src.trainer.fsdp_sft_trainer \
    diffusion.time_reweighting=$REWEIGHT \
    data.train_files=$DATA_ROOT/lingcoder/train.parquet \
    data.val_files=$DATA_ROOT/lingcoder/val.parquet \
    data.max_length=4000 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=$LR \
    data.train_batch_size=576 \
    data.micro_batch_size_per_gpu=6 \
    data.perbatch_cutoff=True \
    data.perbatch_cutoff_type=random_with_input_pad \
    model.partial_pretrain=$CKPT_PATH \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$JOBNAME \
    trainer.project_name=diff-verl \
    trainer.experiment_name=$JOBNAME \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=7 \
    trainer.save_checkpoint_steps=2000 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true