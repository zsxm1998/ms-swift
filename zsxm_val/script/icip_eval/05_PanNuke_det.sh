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
RES_FILE="$LOG_DIR/05_PanNuke_det.log"
mkdir -p $LOG_DIR

# ********************************PanNuke********************************
if [[ $CKPT_DIR == *"-int_"* ]]; then
  QUESTION_FILE=datasets/nucleus_segment/PanNuke/detcls_int_question_PanNuke_part3.jsonl
else
  QUESTION_FILE=datasets/nucleus_segment/PanNuke/detcls_question_PanNuke_part3.jsonl
fi
IMAGE_FOLDER=datasets/nucleus_segment/PanNuke/part3
ANSWER_FILE="$LOG_DIR/z05_PanNuke_part3.jsonl"
VIS_DIR="$LOG_DIR/.vis/05_PanNuke_part3"

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

# Evaluate
echo "----------------------------------PanNuke----------------------------------" > "$RES_FILE"
python zsxm_val/code/eval_path_cv/with_class_detection.py \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --img_dir $IMAGE_FOLDER \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1