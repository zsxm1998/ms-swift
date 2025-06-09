# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
# A2代表只训练第二个vision aligner部分
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift pt \
    --output_dir ./zsxm_checkpoint/nips/0_pretrain/omnipt-qwen2-vl-7b_0608_full_A2_1 \
    --model /c22073/LLM_weights/OmniPT-Qwen2-VL-7B \
    --custom_register_path ./zsxm_model/models/omnipt_qwen2_vl/swift_register.py \
    --deepspeed zero0 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing false \
    --attn_impl flash_attn \
    --freeze_parameters_ratio 1 \
    --trainable_parameters second_visual.merger \
    --dataset ./zsxm_dataset/nips/0_pretrain/align_0608.json \
    --max_length 8192 \
    --max_pixels $((1280*28*28)) \
    --truncation_strategy right \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-3 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false 
