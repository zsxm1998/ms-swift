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
RES_FILE="$LOG_DIR/16_NI_zheyi.log"
mkdir -p $LOG_DIR

# ************************************* Evaluate for nerve detection 0607 *************************************
if [[ $CKPT_DIR == *"-int_"* ]]; then
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0607/NI_det/int_question_test.jsonl
else
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0607/NI_det/question_test.jsonl
fi
IMAGE_FOLDER=datasets/ZheYi0607/NI_det/images
ANSWER_FILE="$LOG_DIR/z16_NI_det_0607.jsonl"
VIS_DIR="$LOG_DIR/.vis/16_NI_det/0607"

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

echo "----------------------------------NI_det 0607----------------------------------" > "$RES_FILE"
python zsxm_val/code/eval_path_cv/no_class_detection.py \
    --img_dir $IMAGE_FOLDER \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1

# ************************************* Evaluate for neural invasion 0607 *************************************
if [[ $CKPT_DIR == *"-int_"* ]]; then
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0607/NI_cls/int_question_test.jsonl
else
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0607/NI_cls/question_test.jsonl
fi
IMAGE_FOLDER=datasets/ZheYi0607/NI_cls/images
ANSWER_FILE="$LOG_DIR/z16_NI_cls_0607.jsonl"

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

echo "----------------------------------neural invasion classification 0607----------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "NI" >> "$RES_FILE" 2>&1



# ************************************* Evaluate for nerve detection 0730 *************************************
if [[ $CKPT_DIR == *"-int_"* ]]; then
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0730/NI_det/int_question_test.jsonl
else
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0730/NI_det/question_test.jsonl
fi
IMAGE_FOLDER=datasets/ZheYi0730/NI_det/images
ANSWER_FILE="$LOG_DIR/z16_NI_det_0730.jsonl"

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

VIS_DIR="$LOG_DIR/.vis/16_NI_det/0730_det_no_class"
echo "----------------------------------NI_det without class 0730----------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/no_class_detection.py \
    --img_dir $IMAGE_FOLDER \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1

VIS_DIR="$LOG_DIR/.vis/16_NI_det/0730_det_with_class"
echo "----------------------------------NI_det with class 0730----------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/with_class_detection.py \
    --img_dir $IMAGE_FOLDER \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1

# ************************************* Evaluate for neural invasion 0730 *************************************
if [[ $CKPT_DIR == *"-int_"* ]]; then
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0730/NI_cls/int_question_test.jsonl
else
  QUESTION_FILE=zsxm_dataset/icip/test/ZheYi0730/NI_cls/question_test.jsonl
fi
IMAGE_FOLDER=datasets/ZheYi0730/NI_cls/images
ANSWER_FILE="$LOG_DIR/z16_NI_cls_0730.jsonl"

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

echo "----------------------------------neural invasion classification 0730----------------------------------" >> "$RES_FILE"
python zsxm_val/code/eval_path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "NI" >> "$RES_FILE" 2>&1
