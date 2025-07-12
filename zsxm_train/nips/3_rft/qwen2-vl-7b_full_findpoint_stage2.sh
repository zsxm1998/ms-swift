# ckpt规则：开始权重#开始权重来源#RFT训练参数
# 数据集总数：8247
# batch_size: per_device_train_batch_size * gradient_accumulation_steps * NPROC_PER_NODE = 2 * 8 * 8 = 128

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=$((1280*28*28)) \
swift rlhf \
    --rlhf_type grpo \
    --model ./zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS1TmNm_G8_bs128_max-token-2048/v0-20250701-173022/checkpoint-510 \
    --output_dir ./zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2TaNa_G16_bs128_max-token-2048 \
    --external_plugins ./zsxm_model/plugin/path_orm.py \
    --reward_funcs pathorm_organ \
    --cosine_min_len_value_wrong 0.0 \
    --cosine_max_len_value_wrong 0.1 \
    --cosine_min_len_value_correct 1.0 \
    --cosine_max_len_value_correct 0.9 \
    --use_vllm true \
    --num_infer_workers 8 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 8192 \
    --sleep_level 0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset ./zsxm_dataset/nips/3_rft/two_stage_find_point/0625_2_thinkall-nothinkall.json \
    --multi_turn_func magnify_wsi \
    --vllm_limit_mm_per_prompt '{"image": 15, "video": 1}' \
    --attn_impl flash_attn \
    --deepspeed zero3 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --max_pixels $((1280*28*28)) \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 125 \
    --save_steps 125 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 0.9 \
    --log_completions true \
    --report_to wandb \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --max_resample_times 3
