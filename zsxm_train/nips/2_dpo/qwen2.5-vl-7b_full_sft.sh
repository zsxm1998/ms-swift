# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
# 此训练是为了通过只训练最后一轮回答，使模型学会通过patch纠正错误回答的能力
# 之前的实验发现DPO会损害性能，因此使用sft继续训练
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --model ./zsxm_checkpoint/nips/1_sft/qwen2.5-vl-7b_0510_full_VAL_2/v0-20250510-021651/checkpoint-9484 \
    --output_dir ./zsxm_checkpoint/nips/2_sft_last_round/qwen2.5-vl-7b_0510_full_VAL_2_v0#sft#full_L_2_lrConst \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_aligner true \
    --dataset ./zsxm_dataset/nips/1_sft/grpo_coldstart_all_rounds.json#256 \
              ./zsxm_dataset/nips/1_sft/grpo_coldstart_last_round.json#256 \
    --max_length 8192 \
    --max_pixels $((1280*28*28)) \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 1 \
    --use_hf false \
    --loss_scale last_round
