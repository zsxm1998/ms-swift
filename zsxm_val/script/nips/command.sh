bash zsxm_val/script/nips/eval/01_thumbnail_seg.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/07_patch_subtyping.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137

bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --think

bash zsxm_val/script/nips/eval/03_nucleus_det_no_class.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --think --func

bash zsxm_val/script/nips/eval/04_nucleus_det_with_class.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --func

bash zsxm_val/script/nips/eval/05_vessel_nerve_lymph_det_seg.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/08_patch_grading.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137

bash zsxm_val/script/nips/eval/06_cancer_in_vessel_nerve_lymph.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/09_classify_nucleus_bbox.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/09_classify_nucleus_bbox.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --think

bash zsxm_val/script/nips/eval/07_patch_subtyping.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --think

bash zsxm_val/script/nips/eval/08_patch_grading.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --think








# --------------------------- 测试使用CVPR的attention提取的patch对缩略图的效果 ---------------------------
bash zsxm_val/script/nips/eval/93_thumbnail_choice_tool.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --func; bash zsxm_val/script/nips/eval/91_thumbnail_choice_additional_tool.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --func

bash zsxm_val/script/nips/eval/94_thumbnail_choice_notool.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137; bash zsxm_val/script/nips/eval/92_thumbnail_choice_additional_notool.sh zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137


# --------------------------- 推理训练集 ---------------------------
python zsxm_val/code/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/08_thumbnail_func_point.json" --answers-file "zsxm_val/results/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048|v0|1137/train_08_thumbnail_func_point_think_10.jsonl" --model-path "zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137" --ignore-origin-system --func --think --seed 10; python zsxm_val/code/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/08_thumbnail_func_point.json" --answers-file "zsxm_val/results/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048|v0|1137/train_08_thumbnail_func_point_10.jsonl" --model-path "zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137" --ignore-origin-system --func --seed 10

python zsxm_val/code/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/02_thumbnail_choice.json" --answers-file "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_07.jsonl" --model-path "zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137" --seed 7; python zsxm_val/code/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/07_private_patch_choice.json" --answers-file "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_07.jsonl" --model-path "zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137" --seed 7


# --------------------- 计算delta ---------------------
python zsxm_val/code/calculate_delta.py --model zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --pivot zsxm_dataset/nips/3_rft/02_thumbnail_choice.json --jsonl "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_09.jsonl" "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_10.jsonl" --output "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_delta_4.jsonl"

python zsxm_val/code/calculate_delta.py --model zsxm_checkpoint/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048/v0-20250706-014450/checkpoint-1137 --pivot zsxm_dataset/nips/3_rft/07_private_patch_choice.json --jsonl "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_08.jsonl" "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_09.jsonl" "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_10.jsonl" --output "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_delta_4.jsonl"
