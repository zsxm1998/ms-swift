#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [--lora-path <LORA_PATH>]"
  exit 1
fi

# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
shift  # Remove CKPT_DIR from arguments

# Initialize LORA_PATH variable
LORA_PATH=""

# Parse optional --lora-path argument
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora-path)
      LORA_PATH="$2"
      shift 2  # Consume both --lora-path and its value
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the argument list for the sub-scripts
ARGS=("$CKPT_DIR")
if [ -n "$LORA_PATH" ]; then
  ARGS+=("--lora-path" "$LORA_PATH")
fi

bash zsxm_val/script/icip_eval/10_TILs.sh "${ARGS[@]}"
bash zsxm_val/script/icip_eval/12_HCC_grading.sh "${ARGS[@]}"
bash zsxm_val/script/icip_eval/13_liver_subtype.sh "${ARGS[@]}"
bash zsxm_val/script/icip_eval/14_ICC_subtype.sh "${ARGS[@]}"
