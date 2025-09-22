model=Dream-org/Dream-Coder-v0-Base-7B

############### Code ###############
tasks="humaneval humaneval_plus"
read -ra TASKS_ARRAY <<< "$tasks"
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_code_results/${TASKS_ARRAY[$i]}-ns0
    echo "Task: ${TASKS_ARRAY[$i]}; Output: $output_path"
    HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch -m lm_eval --model diffllm \
        --model_args pretrained=${model},trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,add_bos_token=true,temperature=0.2,top_p=0.9,alg=entropy \
        --tasks ${TASKS_ARRAY[$i]} \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done

tasks="mbpp mbpp_plus"
read -ra TASKS_ARRAY <<< "$tasks"
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_code_results/${TASKS_ARRAY[$i]}-ns0
    echo "Task: ${TASKS_ARRAY[$i]}; Output: $output_path"
    HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch -m lm_eval --model diffllm \
        --model_args pretrained=${model},trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,add_bos_token=true,temperature=0.1,top_p=0.9,alg=entropy \
        --tasks ${TASKS_ARRAY[$i]} \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done

############### General & Science Multiple Choice###############
tasks="mmlu arc_challenge hellaswag piqa gpqa_main_n_shot winogrande race"
nshots="5 0 0 0 5 5 0"
# tasks="arc_challenge"
# nshots="0"
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_code_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
        --model diffllm \
        --model_args pretrained=$model,add_bos_token=true \
        --tasks ${TASKS_ARRAY[$i]} \
        --batch_size 32 \
        --output_path $output_path \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --log_samples 
done

############### Math ###############
tasks="gsm8k_cot minerva_math"
nshots="8 4"
lengths="256 512"
# tasks="minerva_math"
# nshots="4"
# lengths="512"
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_code_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval --model diffllm \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=0,top_p=0.95 \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples
done