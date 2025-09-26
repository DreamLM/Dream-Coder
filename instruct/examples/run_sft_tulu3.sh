set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: examples/run_sft_tulu3.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m src.trainer.fsdp_sft_trainer \
    diffusion.time_reweighting=linear \
    data.train_files=$HOME/data/tulu3/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.max_length=1024 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=1e-4 \
    data.micro_batch_size=1 \
    model.partial_pretrain=Dream-org/Dream-7B-200B-1e-4 \
    model.trust_remote_code=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=diff-verl \
    trainer.experiment_name=tulu3-sft-diffllm-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true