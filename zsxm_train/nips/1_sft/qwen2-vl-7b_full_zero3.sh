# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
# 本模型attn_impl=flash_attn会有bug 【RuntimeError: cu_seqlens_q must be on CUDA】，所以用sdpa
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --output_dir ./zsxm_checkpoint/nips/1_sft/qwen2-vl-7b_0306_full_VAL_2 \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing true \
    --attn_impl sdpa \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset ./zsxm_dataset/nips/1_sft/SFT-0306.json \
              swift/self-cognition#500 \
    --max_length 8192 \
    --max_pixels $((336*336)) \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false \
    --model_author "浙江大学VIPA实验室" "Zhejiang University VIPA Laboratory" \
    --model_name "OmniPT" "OmniPT"
