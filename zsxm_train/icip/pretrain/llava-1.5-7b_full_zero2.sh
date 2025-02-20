# ckpt命名规则：训练类型/模型_数据集_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift pt \
    --output_dir ./zsxm_checkpoint/icip/pretrain/llava-1.5-7b_2404181_t-A_e-1 \
    --model swift/llava-1.5-7b-hf \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing false \
    --attn_impl flash_attn \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false \
    --dataset ./zsxm_dataset/icip/pretrain/2404181_quilt1m_pathcapdiff_pathinstructP1.json \
    --max_length 4096 \
    --max_pixels $((336*336)) \
    --truncation_strategy right \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-3 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false
