#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng] [--lora-path <LORA_PATH>]"
  exit 1
fi

# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
shift  # Remove CKPT_DIR from arguments

# Initialize variables
LORA_PATH="None"
NG_ARG=""

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora-path)
      LORA_PATH="$2"
      shift 2  # Consume both --lora-path and its value
      ;;
    -ng)
      NG_ARG="-ng"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if the directory exists locally
if [ -d "$CKPT_DIR" ]; then
  CKPT_NAME=$(echo "$CKPT_DIR" | awk -F'/' '{print $(NF-3)"/"$(NF-2)}')
else
  CKPT_NAME="baseline/$(basename "$CKPT_DIR")"
fi

LOG_DIR="zsxm_val/icip/$CKPT_NAME"
RES_FILE="$LOG_DIR/99_language_benchmark.log"
mkdir -p $LOG_DIR


# -------------------------------------------------- PVQA --------------------------------------------------
QUESTION_FILE=datasets/PathVQA/pvqa_test_wo_ans.jsonl
IMAGE_FOLDER=datasets/PathVQA/images/test
ANSWER_FILE="$LOG_DIR/z99_PathVQA.jsonl"

# Build Python arguments
PYTHON_ARGS=(
  --model-path "$CKPT_DIR"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --answers-file "$ANSWER_FILE"
)

# Add --lora-path only if it's not "None"
if [ "$LORA_PATH" != "None" ]; then
  PYTHON_ARGS+=(--lora-path "$LORA_PATH")
fi

# Run Python script if -ng is not present
if [[ -z "$NG_ARG" ]]; then
  python zsxm_val/code/model_vqa.py "${PYTHON_ARGS[@]}"
fi

echo "------------------------------PathVQA------------------------------" > "$RES_FILE"
python zsxm_val/code/eval_path_benchmark/quilt_eval.py \
    --gt datasets/PathVQA/pvqa_test_w_ans.json \
    --pred $ANSWER_FILE >> "$RES_FILE" 2>&1


# -------------------------------------------------- PMCVQA --------------------------------------------------
QUESTION_FILE=datasets/PMC-VQA/pmcvqa_test_wo_ans.json
IMAGE_FOLDER=datasets/PMC-VQA/images
ANSWER_FILE="$LOG_DIR/z99_PMC-VQA.jsonl"
python zsxm_val/code/model_vqa_science.py \
    --model-path $CKPT_DIR \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $ANSWER_FILE \
    --single-pred-prompt

echo -e "\n------------------------------PMC-VQA------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_benchmark/pmc_eval.py \
    --question-file $QUESTION_FILE \
    --result-file $ANSWER_FILE \
    --output-file "$LOG_DIR/z99_PMC-VQA_output.jsonl" \
    --output-result "$LOG_DIR/z99_PMC-VQA_result.json" >> "$RES_FILE" 2>&1


# -------------------------------------------------- QUILT-VQA --------------------------------------------------
QUESTION_FILE=datasets/QuiltVQA/quiltvqa_test_wo_ans.jsonl
IMAGE_FOLDER=datasets/QuiltVQA/images
ANSWER_FILE="$LOG_DIR/z99_QuiltVQA.jsonl"

# Build Python arguments
PYTHON_ARGS=(
  --model-path "$CKPT_DIR"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --answers-file "$ANSWER_FILE"
)

# Add --lora-path only if it's not "None"
if [ "$LORA_PATH" != "None" ]; then
  PYTHON_ARGS+=(--lora-path "$LORA_PATH")
fi

# Run Python script if -ng is not present
if [[ -z "$NG_ARG" ]]; then
  python zsxm_val/code/model_vqa.py "${PYTHON_ARGS[@]}"
fi

echo -e "\n------------------------------QUILT-VQA------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_benchmark/quilt_eval.py \
    --quilt True \
    --gt datasets/QuiltVQA/quiltvqa_test_w_ans.json \
    --pred $ANSWER_FILE >> "$RES_FILE" 2>&1


# -------------------------------------------------- QUILT-VQA RED --------------------------------------------------
QUESTION_FILE=datasets/QuiltVQA/quiltvqa_red_test_wo_ans.jsonl
IMAGE_FOLDER=datasets/QuiltVQA/red_circle
ANSWER_FILE="$LOG_DIR/z99_QuiltVQA-red-circle.jsonl"

# Build Python arguments
PYTHON_ARGS=(
  --model-path "$CKPT_DIR"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --answers-file "$ANSWER_FILE"
)

# Add --lora-path only if it's not "None"
if [ "$LORA_PATH" != "None" ]; then
  PYTHON_ARGS+=(--lora-path "$LORA_PATH")
fi

# Run Python script if -ng is not present
if [[ -z "$NG_ARG" ]]; then
  python zsxm_val/code/model_vqa.py "${PYTHON_ARGS[@]}"
fi

echo -e "\n------------------------------Quilt-VQA Red Circle------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_benchmark/quilt_eval.py \
    --quilt True \
    --gt datasets/QuiltVQA/quiltvqa_red_test_w_ans.json \
    --pred $ANSWER_FILE >> "$RES_FILE" 2>&1


# QUILT-VQA No RED
QUESTION_FILE=datasets/QuiltVQA/quiltvqa_nored_test_wo_ans.jsonl
IMAGE_FOLDER=datasets/QuiltVQA/images
ANSWER_FILE="$LOG_DIR/z99_QuiltVQA-nored-circle.jsonl"

# Build Python arguments
PYTHON_ARGS=(
  --model-path "$CKPT_DIR"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --answers-file "$ANSWER_FILE"
)

# Add --lora-path only if it's not "None"
if [ "$LORA_PATH" != "None" ]; then
  PYTHON_ARGS+=(--lora-path "$LORA_PATH")
fi

# Run Python script if -ng is not present
if [[ -z "$NG_ARG" ]]; then
  python zsxm_val/code/model_vqa.py "${PYTHON_ARGS[@]}"
fi

echo -e "\n------------------------------Quilt-VQA No Red Circle------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_benchmark/quilt_eval.py \
    --quilt True \
    --gt ./datasets/QuiltVQA/quiltvqa_red_test_w_ans.json \
    --pred $ANSWER_FILE >> "$RES_FILE" 2>&1
