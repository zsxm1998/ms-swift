# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type dpo \
    --model ./zsxm_checkpoint/nips/1_sft/qwen2.5-vl-7b_0415_full_VAL_2/v0-20250416-054620/checkpoint-9444 \
    --output_dir ./zsxm_checkpoint/nips/2_dpo/DPO_qwen2.5-vl-7b_0415_full_VAL_2_v0#sft#lora_VAL_1 \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --train_type lora \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset ./zsxm_dataset/nips/2_dpo/liver_thumbnail.json \
              ./zsxm_dataset/nips/2_dpo/function_liver_thumbnail.json \
              ./zsxm_dataset/nips/2_dpo/liver_patch.json \
              ./zsxm_dataset/nips/2_dpo/rft_seg_det_cold_start.json \
    --max_length 8192 \
    --max_pixels $((1280*28*28)) \
    --truncation_strategy right \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false
