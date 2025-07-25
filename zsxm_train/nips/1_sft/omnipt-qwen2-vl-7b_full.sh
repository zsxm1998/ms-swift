# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --output_dir ./zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_11031_full_VAL_2 \
    --model zsxm_checkpoint/nips/0_pretrain/omnipt-qwen2-vl-7b_0608_full_A2_1/v0-20250609-003645/checkpoint-2533 \
    --custom_register_path ./zsxm_model/models/omnipt_qwen2_vl/swift_register.py \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_aligner false \
    --trainable_parameters visual \
    --dataset ./zsxm_dataset/nips/1_sft/CVPR_11031.json \
              swift/self-cognition#500 \
    --max_length 8192 \
    --max_pixels $((1280*28*28)) \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false \
    --model_author "浙江大学VIPA实验室" "Zhejiang University VIPA Laboratory" \
    --model_name "OmniPT" "OmniPT"
