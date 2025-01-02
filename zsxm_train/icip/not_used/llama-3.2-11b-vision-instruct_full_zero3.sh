# ckpt命名规则：训练类型/模型_数据集_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --output_dir ./zsxm_checkpoint/icip/full/llama-3.2-11b-vision-instruct_2412230-int_t-VAL_e-2 \
    --model LLM-Research/Llama-3.2-11B-Vision-Instruct \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing true \
    --attn_impl sdpa \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset ./zsxm_dataset/icip/finetune/2412230_int_patho-instruct_patho-vision.json \
              swift/self-cognition#1000 \
    --max_length 4096 \
    --max_pixels $((512*512)) \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 1 \
    --model_author "VIPA实验室" "vipa-lab" \
    --model_name "VIPA病理助手" "VIPA-Path-Assistant" \
    --system "你是VIPA病理助手，英文名是VIPA-Path-Assistant，是由浙江大学VIPA实验室开发的专注于病理学的多模态大语言模型。"
    --use_hf false
