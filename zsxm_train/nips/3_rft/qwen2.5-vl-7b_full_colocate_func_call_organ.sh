# One GPU is left for vLLM inference acceleration. GPU memory: 8 * 80GiB
# ckpt规则：开始权重#开始权重来源#RFT训练参数
# 数据集总数：1798+4500+2000+2000+2494+326+4500+5000=22618
# batch_size: per_device_train_batch_size * gradient_accumulation_steps * NPROC_PER_NODE = 2 * 8 * 8 = 128

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=$((1280*28*28)) \
swift rlhf \
    --rlhf_type grpo \
    --model ./zsxm_checkpoint/nips/2_sft_last_round/qwen2.5-vl-7b_0510_full_VAL_2_v0#sft#full_L_2/v0-20250513-015709/checkpoint-8 \
    --output_dir ./zsxm_checkpoint/nips/3_rft/qwen2.5-vl-7b_0510_full_VAL_2_v0#last_round_sft#ToolCall_organ-reward_full_bs128_temp0.9 \
    --external_plugins ./zsxm_model/plugin/path_orm.py \
    --reward_funcs pathorm_organ \
    --cosine_min_len_value_wrong 0.0 \
    --cosine_max_len_value_wrong 0.1 \
    --cosine_min_len_value_correct 1.0 \
    --cosine_max_len_value_correct 1.0 \
    --use_vllm true \
    --num_infer_workers 8 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 8192 \
    --sleep_level 0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset ./zsxm_dataset/nips/3_rft/01_thumbnail_seg.json \
              ./zsxm_dataset/nips/3_rft/02_thumbnail_choice.json#4500 \
              ./zsxm_dataset/nips/3_rft/03_nucleus_det_no_class.json#2000 \
              ./zsxm_dataset/nips/3_rft/04_nucleus_det_with_class.json \
              ./zsxm_dataset/nips/3_rft/05_structure_seg_det.json \
              ./zsxm_dataset/nips/3_rft/06_cancer_in_structure.json \
              ./zsxm_dataset/nips/3_rft/07_private_patch_choice.json#4500 \
              ./zsxm_dataset/nips/3_rft/08_thumbnail_func_point.json#5000 \
    --multi_turn_func magnify_wsi \
    --vllm_limit_mm_per_prompt '{"image": 7, "video": 1}' \
    --attn_impl flash_attn \
    --deepspeed zero3 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --max_pixels $((1280*28*28)) \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 125 \
    --save_steps 125 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --log_completions true \
    --report_to wandb \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --max_resample_times 3
