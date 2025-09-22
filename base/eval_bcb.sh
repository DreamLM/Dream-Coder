export LC_ALL="POSIX"

INPUT_MODEL=Dream-org/Dream-Coder-v0-Base-7B
OUTPUT_DIR=evals_code_results

export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


subset=hard
temperature=0.1
max_new_tokens=512
top_p=0.95
alg='maskgit_plus'
alg_temp=0
tag=Temp_${temperature}-Len_${max_new_tokens}-Topp_${top_p}-Alg_${alg}_${alg_temp}
# split: $subset
mkdir -p ${OUTPUT_DIR}/bigcodebench/$subset
python bigcodebench/generate.py \
    --model ${INPUT_MODEL} \
    --split complete \
    --subset $subset \
    --greedy  \
    --bs 1 \
    --n_samples 1 \
    --resume  \
    --backend dllm \
    --save_path ${OUTPUT_DIR}/bigcodebench/$subset/completion-$tag.jsonl \
    --temperature $temperature \
    --max_new_tokens $max_new_tokens \
    --steps $max_new_tokens \
    --top_p $top_p \
    --alg $alg \
    --alg_temp $alg_temp

python bigcodebench/sanitize.py \
    --samples ${OUTPUT_DIR}/bigcodebench/$subset/completion-$tag.jsonl \
    --calibrate

python bigcodebench/evaluate.py \
    --split complete \
    --subset $subset \
    --no-gt \
    --samples ${OUTPUT_DIR}/bigcodebench/$subset/completion-$tag-sanitized-calibrated.jsonl


subset=full
temperature=0.1
max_new_tokens=512
top_p=0.95
alg='entropy'
alg_temp=0
tag=Temp_${temperature}-Len_${max_new_tokens}-Topp_${top_p}-Alg_${alg}_${alg_temp}
# split: $subset
mkdir -p ${OUTPUT_DIR}/bigcodebench/$subset
python bigcodebench/generate.py \
    --model ${INPUT_MODEL} \
    --split complete \
    --subset $subset \
    --greedy  \
    --bs 1 \
    --n_samples 1 \
    --resume  \
    --backend dllm \
    --save_path ${OUTPUT_DIR}/bigcodebench/$subset/completion-$tag.jsonl \
    --temperature $temperature \
    --max_new_tokens $max_new_tokens \
    --steps $max_new_tokens \
    --top_p $top_p \
    --alg $alg \
    --alg_temp $alg_temp

python bigcodebench/sanitize.py \
    --samples ${OUTPUT_DIR}/bigcodebench/$subset/completion-$tag.jsonl \
    --calibrate

python bigcodebench/evaluate.py \
    --split complete \
    --subset $subset \
    --no-gt \
    --samples ${OUTPUT_DIR}/bigcodebench/$subset/completion-$tag-sanitized-calibrated.jsonl
