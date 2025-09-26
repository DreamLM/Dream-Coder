WORKING_DIR=$PWD
CRUX_ROOT_DIR=code_eval/cruxeval
MODEL=${1}
TYPE=${2:-output_cot}  # output_direct, output_cot, input_direct, input_cot
CONFIG_NAME=${3:-cruxeval}
PORT=${4:-29500}

arrTYPE=(${TYPE//_/ })
MODE=${arrTYPE[0]}
MODELNAME="${MODEL,,}"

echo "Generating $TYPE for $MODEL"

python $CRUX_ROOT_DIR/prepare_cruxeval.py \
    --output_path data/cruxeval/$TYPE.jsonl \
    --prompt_types $TYPE

PYTHONPATH=. accelerate launch --main_process_port $PORT scripts/generate.py \
    --input data/cruxeval/$TYPE.jsonl \
    --output res/cruxeval_bs1/${MODELNAME}_${TYPE}_${CONFIG_NAME}.jsonl \
    --model $MODEL \
    --config_name ${CONFIG_NAME} \
    --batch_size 1

PYTHONPATH=. python scripts/postprocess.py \
    --input res/cruxeval_bs1/${MODELNAME}_${TYPE}_${CONFIG_NAME}.jsonl \
    --output res/cruxeval_bs1/${MODELNAME}_${TYPE}_${CONFIG_NAME}.p.json \
    --type cruxeval

cd $CRUX_ROOT_DIR/evaluation
python evaluate_generations.py \
    --generations_path $WORKING_DIR/res/cruxeval_bs1/${MODELNAME}_${TYPE}_${CONFIG_NAME}.p.json \
    --scored_results_path $WORKING_DIR/res/cruxeval_bs1/${MODELNAME}_${TYPE}_${CONFIG_NAME}.s.json \
    --mode $MODE
