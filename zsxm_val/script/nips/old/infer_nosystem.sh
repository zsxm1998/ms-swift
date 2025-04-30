CUDA_VISIBLE_DEVICES=2 \
MAX_PIXELS=$((336*336)) \
swift infer \
    --infer_backend vllm \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 8192 \
    --limit_mm_per_prompt '{"image": 10, "video": 10}' \
    --max_batch_size 16 \
    --tensor_parallel_size 8 \
    --model /c22073/codes/ms-swift/zsxm_checkpoint/nips/1_sft/llava-1.5-7b_0306_full_VAL_2/v0-20250309-202547/checkpoint-9030 \
    --val_dataset /c22073/codes/ms-swift/zsxm_dataset/nips/9_test/02_thumbnail_choice.json \
    --result_path /c22073/codes/ms-swift/zsxm_val/nips/llava_thumb_result_withsystem.jsonl