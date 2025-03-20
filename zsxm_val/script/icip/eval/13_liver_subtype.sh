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
RES_FILE="$LOG_DIR/13_liver_subtype.log"
mkdir -p $LOG_DIR

# ************************************* Evaluate for Patch *************************************
QUESTION_FILE=datasets/ZheYi0607/liver_cancer/patch_question_test.jsonl
IMAGE_FOLDER=datasets/ZheYi0607/liver_cancer/patches
ANSWER_FILE="$LOG_DIR/z13_liver_subtype.jsonl"

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

echo -e "------------------------------Patch pathological type open ended classification------------------------------" > "$RES_FILE"
python zsxm_val/code/eval_path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "liver_subtype_patho" >> "$RES_FILE" 2>&1

echo -e "\n------------------------------Patch pathological type close ended classification------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/choice_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "liver_subtype_patho" >> "$RES_FILE" 2>&1

echo -e "\n------------------------------Patch ICC grading close ended classification------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/choice_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "liver_subtype_ICC_grading" >> "$RES_FILE" 2>&1

echo -e "\n------------------------------Patch HCC subtype close ended classification------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/choice_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "liver_subtype_HCC_subtype" >> "$RES_FILE" 2>&1
