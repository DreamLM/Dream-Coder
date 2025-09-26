for model in "ByteDance-Seed/Seed-Coder-8B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct"; do
    for dataset in "humaneval" "mbpp"; do
        CUDA_VISIBLE_DEVICES=4,5,6,7 evalplus.evaluate \
            --model $model \
            --dataset $dataset \
            --backend vllm \
            --tp 4 \
            --greedy
    done
done

for dataset in "humaneval" "mbpp"; do
    evalplus.evaluate \
        --model inception/mercury-coder-small-beta \
        --dataset $dataset \
        --backend openai \
        --greedy
done

for dataset in "humaneval" "mbpp"; do
    evalplus.evaluate \
        --model google/gemini-2.0-flash-001 \
        --dataset $dataset \
        --backend openai \
        --greedy
done

for dataset in "humaneval" "mbpp"; do
    evalplus.evaluate \
        --model google/gemma-3-12b-it \
        --dataset $dataset \
        --backend openai \
        --greedy
done

# for dataset in "humaneval" "mbpp"; do
#     evalplus.evaluate \
#         --model meta-llama/llama-3.1-8b-instruct \
#         --dataset $dataset \
#         --backend openai \
#         --greedy
# done

