# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
# 这个脚本zero3不能变，用zero2就爆显存
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --output_dir ./zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2 \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --dataset ./zsxm_dataset/nips/1_sft/SFT-0415.json \
              swift/self-cognition#500 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --train_type custom \
    --optimizer custom \
    --external_plugins 'examples/train/multimodal/lora_llm_full_vit/custom_plugin.py' \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_checkpointing true \
    --max_length 8192 \
    --max_pixels $((1280*28*28)) \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false \
    --model_author "浙江大学VIPA实验室" "Zhejiang University VIPA Laboratory" \
    --model_name "OmniPT" "OmniPT"
