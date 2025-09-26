#!/bin/bash

# 高级并行执行eval脚本，支持彩色输出区分和中断处理
# 使用方法: ./parallel_eval_advanced.sh [options]

# 默认参数
MAX_PARALLEL=3
VERBOSE=false
OUTPUT_MODE="mixed"  # mixed, separate, quiet
SHOW_TIMESTAMPS=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 为不同任务分配颜色
TASK_COLORS=("${RED}" "${GREEN}" "${YELLOW}" "${BLUE}" "${PURPLE}" "${CYAN}")

# 显示帮助信息
show_help() {
    echo "高级并行eval执行脚本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -j, --jobs N         最大并行任务数 (默认: 3)"
    echo "  -v, --verbose        显示实时输出"
    echo "  -o, --output MODE    输出模式:"
    echo "                       mixed: 混合显示所有任务输出 (默认)"
    echo "                       separate: 分别显示每个任务输出"
    echo "                       quiet: 仅显示状态信息"
    echo "  -t, --timestamps     显示时间戳"
    echo "  -h, --help           显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -j 4 -v              # 4个并行任务，显示实时输出"
    echo "  $0 --output separate    # 分别显示每个任务的输出"
    echo "  $0 -v -t                # 显示实时输出和时间戳"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -o|--output)
            OUTPUT_MODE="$2"
            shift 2
            ;;
        -t|--timestamps)
            SHOW_TIMESTAMPS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证参数
if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL" -lt 1 ]; then
    echo "错误: 并行任务数必须是正整数"
    exit 1
fi

if [[ ! "$OUTPUT_MODE" =~ ^(mixed|separate|quiet)$ ]]; then
    echo "错误: 输出模式必须是 mixed, separate 或 quiet"
    exit 1
fi

# 定义要执行的命令数组
declare -a EVAL_COMMANDS=(
    "CUDA_VISIBLE_DEVICES=4 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/lingcoder_fixlower_3epoch_cart_randomcutoff_code-960k-ctx4k_bs576_inputpad_lr5e-6/global_step_26652 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    "CUDA_VISIBLE_DEVICES=5 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_5000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    "CUDA_VISIBLE_DEVICES=6 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_10000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
    "CUDA_VISIBLE_DEVICES=7 evalplus.evaluate --model /home/ndfl4zki/ndfl4zkiuser04/codes_posttrain/diffllm-verl/ckpts/mixturecodev2_cart_randomcutoff_code-960k-ctx4k_bs288_wd0.1_ct_inputpad_lr2e-6/global_step_15000 --trust_remote_code True --max_new_tokens 768 --diffusion_steps 768 --dataset humaneval --backend dllm --temperature 0.1"
)

# 全局变量
declare -a RUNNING_PIDS=()
declare -a COMPLETED_JOBS=()
declare -a FAILED_JOBS=()
INTERRUPTED=false
TOTAL_JOBS=${#EVAL_COMMANDS[@]}
COMPLETED_COUNT=0
FAILED_COUNT=0

# 获取时间戳
get_timestamp() {
    if [ "$SHOW_TIMESTAMPS" = true ]; then
        date '+[%H:%M:%S]'
    fi
}

# 信号处理函数
cleanup() {
    echo -e "\n${YELLOW}$(get_timestamp) 收到中断信号，正在停止剩余任务...${NC}"
    INTERRUPTED=true

    # 杀死所有正在运行的进程
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}$(get_timestamp) 正在停止进程 $pid...${NC}"
            kill -TERM "$pid" 2>/dev/null
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null
            fi
        fi
    done

    wait
    print_summary
    exit 130
}

# 绑定信号处理
trap cleanup SIGINT SIGTERM

# 打印进度信息
print_progress() {
    local running_count=${#RUNNING_PIDS[@]}
    echo -e "${BLUE}$(get_timestamp) 进度: ${COMPLETED_COUNT}/${TOTAL_JOBS} 完成, ${FAILED_COUNT} 失败, ${running_count} 运行中${NC}"
}

# 打印总结信息
print_summary() {
    echo -e "\n${BLUE}$(get_timestamp) === 执行总结 ===${NC}"
    echo -e "${GREEN}成功完成: ${COMPLETED_COUNT}/${TOTAL_JOBS}${NC}"
    echo -e "${RED}失败任务: ${FAILED_COUNT}${NC}"
    if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
        echo -e "${RED}失败的任务:${NC}"
        for job in "${FAILED_JOBS[@]}"; do
            echo -e "${RED}  - $job${NC}"
        done
    fi
}

# 获取模型名称
get_model_name() {
    local cmd="$1"
    echo "$cmd" | grep -o 'global_step_[0-9]*' | head -1
}

# 获取GPU设备号
get_gpu_device() {
    local cmd="$1"
    echo "$cmd" | grep -o 'CUDA_VISIBLE_DEVICES=[0-9]*' | cut -d'=' -f2
}

# 获取任务颜色
get_task_color() {
    local job_index=$1
    local color_index=$((job_index % ${#TASK_COLORS[@]}))
    echo "${TASK_COLORS[$color_index]}"
}

# 执行单个任务
run_job() {
    local job_index=$1
    local cmd="${EVAL_COMMANDS[$job_index]}"
    local model_name=$(get_model_name "$cmd")
    local gpu_device=$(get_gpu_device "$cmd")
    local log_file="eval_log_${model_name}_$$_${job_index}.log"
    local task_color=$(get_task_color "$job_index")
    local task_prefix="${task_color}[任务$((job_index + 1))-GPU${gpu_device}]${NC}"

    echo -e "${YELLOW}$(get_timestamp) 开始任务 $((job_index + 1)): $model_name (GPU $gpu_device)${NC}"

    case "$OUTPUT_MODE" in
        "mixed")
            if [ "$VERBOSE" = true ]; then
                {
                    eval "$cmd" 2>&1 | while IFS= read -r line; do
                        echo -e "$(get_timestamp) $task_prefix $line"
                    done
                } | tee "$log_file"
                local exit_code=${PIPESTATUS[0]}
            else
                eval "$cmd" > "$log_file" 2>&1
                local exit_code=$?
            fi
            ;;
        "separate")
            echo -e "${task_color}$(get_timestamp) === 任务 $((job_index + 1)) 输出开始 ===${NC}"
            eval "$cmd" 2>&1 | tee "$log_file" | while IFS= read -r line; do
                echo -e "$task_prefix $line"
            done
            echo -e "${task_color}$(get_timestamp) === 任务 $((job_index + 1)) 输出结束 ===${NC}"
            local exit_code=${PIPESTATUS[0]}
            ;;
        "quiet")
            eval "$cmd" > "$log_file" 2>&1
            local exit_code=$?
            ;;
    esac

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}$(get_timestamp) 任务 $((job_index + 1)) 完成: $model_name (GPU $gpu_device)${NC}"
        COMPLETED_JOBS+=("$model_name")
        ((COMPLETED_COUNT++))
    else
        echo -e "${RED}$(get_timestamp) 任务 $((job_index + 1)) 失败: $model_name (GPU $gpu_device) (退出代码: $exit_code)${NC}"
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
    echo -e "${BLUE}$(get_timestamp) 开始并行执行 ${TOTAL_JOBS} 个eval任务，最大并行数: ${MAX_PARALLEL}${NC}"
    echo -e "${BLUE}$(get_timestamp) 输出模式: ${OUTPUT_MODE}${NC}"
    echo -e "${BLUE}$(get_timestamp) 实时输出: $([ "$VERBOSE" = true ] && echo "开启" || echo "关闭")${NC}"
    echo -e "${BLUE}$(get_timestamp) 时间戳: $([ "$SHOW_TIMESTAMPS" = true ] && echo "开启" || echo "关闭")${NC}"
    echo -e "${BLUE}$(get_timestamp) 日志文件将保存在当前目录中${NC}"
    echo ""

    local job_index=0

    while [ $job_index -lt $TOTAL_JOBS ] && [ "$INTERRUPTED" = false ]; do
        cleanup_finished_jobs

        if [ ${#RUNNING_PIDS[@]} -lt $MAX_PARALLEL ]; then
            run_job $job_index &
            local new_pid=$!
            RUNNING_PIDS+=("$new_pid")
            ((job_index++))
        else
            sleep 1
        fi

        # 在mixed模式且verbose关闭时显示进度
        if [ "$OUTPUT_MODE" = "mixed" ] && [ "$VERBOSE" = false ]; then
            print_progress
        elif [ "$OUTPUT_MODE" = "quiet" ]; then
            print_progress
        fi
    done

    if [ "$INTERRUPTED" = false ]; then
        echo -e "\n${YELLOW}$(get_timestamp) 所有任务已启动，等待完成...${NC}"
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

echo -e "\n${GREEN}$(get_timestamp) 脚本执行完成${NC}"