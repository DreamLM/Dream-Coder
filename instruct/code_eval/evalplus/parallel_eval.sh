#!/bin/bash

# 并行执行eval脚本，支持中断处理
# 使用方法: ./parallel_eval.sh [max_parallel_jobs]

# 设置默认的最大并行任务数
MAX_PARALLEL=${1:-3}

# 定义要执行的命令数组
declare -a EVAL_COMMANDS=(
    # "CUDA_VISIBLE_DEVICES=4 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/lingcoder_fixlower_3epoch_cart_randomcutoff_code-960k-ctx4k_bs576_inputpad_lr5e-6/global_step_26652 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    # "CUDA_VISIBLE_DEVICES=5 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_5000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    # "CUDA_VISIBLE_DEVICES=6 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_10000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    # "CUDA_VISIBLE_DEVICES=7 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_15000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"

    "CUDA_VISIBLE_DEVICES=4 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_20000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    "CUDA_VISIBLE_DEVICES=5 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_25000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    "CUDA_VISIBLE_DEVICES=6 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_18226 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
)

# 全局变量
declare -a RUNNING_PIDS=()
declare -a COMPLETED_JOBS=()
declare -a FAILED_JOBS=()
INTERRUPTED=false
TOTAL_JOBS=${#EVAL_COMMANDS[@]}
COMPLETED_COUNT=0
FAILED_COUNT=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 信号处理函数
cleanup() {
    echo -e "\n${YELLOW}收到中断信号，正在停止剩余任务...${NC}"
    INTERRUPTED=true

    # 杀死所有正在运行的进程
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}正在停止进程 $pid...${NC}"
            kill -TERM "$pid" 2>/dev/null
            # 给进程一些时间优雅退出
            sleep 2
            # 如果还在运行，强制杀死
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null
            fi
        fi
    done

    # 等待所有子进程结束
    wait

    print_summary
    exit 130
}

# 绑定信号处理
trap cleanup SIGINT SIGTERM

# 打印进度信息
print_progress() {
    local running_count=${#RUNNING_PIDS[@]}
    echo -e "${BLUE}进度: ${COMPLETED_COUNT}/${TOTAL_JOBS} 完成, ${FAILED_COUNT} 失败, ${running_count} 运行中${NC}"
}

# 打印总结信息
print_summary() {
    echo -e "\n${BLUE}=== 执行总结 ===${NC}"
    echo -e "${GREEN}成功完成: ${COMPLETED_COUNT}/${TOTAL_JOBS}${NC}"
    echo -e "${RED}失败任务: ${FAILED_COUNT}${NC}"
    if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
        echo -e "${RED}失败的任务:${NC}"
        for job in "${FAILED_JOBS[@]}"; do
            echo -e "${RED}  - $job${NC}"
        done
    fi
}

# 获取模型名称（用于日志文件命名）
get_model_name() {
    local cmd="$1"
    echo "$cmd" | grep -o 'global_step_[0-9]*' | head -1
}

# 执行单个任务
run_job() {
    local job_index=$1
    local cmd="${EVAL_COMMANDS[$job_index]}"
    local model_name=$(get_model_name "$cmd")
    local log_file="eval_log_${model_name}_$$_${job_index}.log"

    echo -e "${YELLOW}开始任务 $((job_index + 1)): $model_name${NC}"

    # 执行命令并捕获输出
    eval "$cmd" > "$log_file" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}任务 $((job_index + 1)) 完成: $model_name${NC}"
        COMPLETED_JOBS+=("$model_name")
        ((COMPLETED_COUNT++))
    else
        echo -e "${RED}任务 $((job_index + 1)) 失败: $model_name (退出代码: $exit_code)${NC}"
        FAILED_JOBS+=("$model_name")
        ((FAILED_COUNT++))
    fi

    return $exit_code
}

# 清理完成的进程
cleanup_finished_jobs() {
    local new_running_pids=()
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_running_pids+=("$pid")
        fi
    done
    RUNNING_PIDS=("${new_running_pids[@]}")
}

# 主执行函数
main() {
    echo -e "${BLUE}开始并行执行 ${TOTAL_JOBS} 个eval任务，最大并行数: ${MAX_PARALLEL}${NC}"
    echo -e "${BLUE}日志文件将保存在当前目录中${NC}\n"

    local job_index=0

    while [ $job_index -lt $TOTAL_JOBS ] && [ "$INTERRUPTED" = false ]; do
        # 清理已完成的任务
        cleanup_finished_jobs

        # 如果当前运行的任务数小于最大并行数，启动新任务
        if [ ${#RUNNING_PIDS[@]} -lt $MAX_PARALLEL ]; then
            # 在后台启动新任务
            run_job $job_index &
            local new_pid=$!
            RUNNING_PIDS+=("$new_pid")
            ((job_index++))
        else
            # 等待一段时间再检查
            sleep 1
        fi

        print_progress
    done

    # 等待所有剩余任务完成
    if [ "$INTERRUPTED" = false ]; then
        echo -e "\n${YELLOW}所有任务已启动，等待完成...${NC}"
        for pid in "${RUNNING_PIDS[@]}"; do
            wait "$pid"
        done
    fi

    print_summary
}

# 检查必要的环境
if ! command -v evalplus.evaluate &> /dev/null; then
    echo -e "${RED}错误: evalplus.evaluate 命令未找到，请确保已正确安装 evalplus${NC}"
    exit 1
fi

# 运行主程序
main

echo -e "\n${GREEN}脚本执行完成${NC}"