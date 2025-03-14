# pipeline_parallel_size设置为2时推理速度显著慢于1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=112896 \
swift app \
    --model zsxm_checkpoint/nips/1_sft/qwen2-vl-7b_0306_full_VAL_2/v0-20250307-021757/checkpoint-9022 \
    --served_model_name checkpoint-9022 \
    --studio_title OmniPT-Qwen2-VL-7B \
    --stream true \
    --lang zh \
    --infer_backend vllm \
    --gpu_memory_utilization 0.95 \
    --max_model_len 8192 \
    --max_new_tokens 8192 \
    --limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --pipeline_parallel_size 1 \
    --tensor_parallel_size 4 \
    --temperature 1 \
    --server_port 8888
    