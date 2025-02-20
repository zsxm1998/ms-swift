# ckpt命名规则：训练类型/模型_数据集_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --output_dir ./zsxm_checkpoint/icip/lora/internvl2.5-8b_2412230-int_t-AL_e-2 \
    --model OpenGVLab/InternVL2_5-8B \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --train_type lora \
    --gradient_checkpointing false \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_aligner false \
    --dataset ./zsxm_dataset/icip/finetune/2412230_int_patho-instruct_patho-vision.json \
              swift/self-cognition#1000 \
    --max_length 4096 \
    --max_pixels $((448*448)) \
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
    --system "你是VIPA病理助手，英文名是VIPA-Path-Assistant，是由浙江大学VIPA实验室开发的专注于病理学的多模态大语言模型。" \
    --use_hf false
