# ckpt命名规则：训练类型/模型_数据集_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="4,5,6,7" \
NPROC_PER_NODE=4 \
swift sft \
    --output_dir ./zsxm_checkpoint/icip/qlora/qwen2-vl-72b_2412230-int_t-VAL_e-2 \
    --model Qwen/Qwen2-VL-72B-Instruct \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --train_type lora \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_storage bfloat16 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset ./zsxm_dataset/icip/finetune/2412230_int_patho-instruct_patho-vision.json \
              swift/self-cognition#1000 \
    --max_length 4096 \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 1 \
    --model_author "VIPA实验室" "vipa-lab" \
    --model_name "VIPA病理助手" "VIPA-Path-Assistant" \
    --use_hf false
