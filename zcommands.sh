# python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/01_thumbnail_seg.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/01_thumbnail_seg.jsonl -tp 8

# python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/02_thumbnail_choice.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/02_thumbnail_choice.jsonl -tp 8

# python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/03_nucleus_det_no_class.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/03_nucleus_det_no_class.jsonl -tp 8

python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/04_nucleus_det_with_class.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/04_nucleus_det_with_class.jsonl -tp 8

python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/05_classify_nucleus_bbox.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/05_classify_nucleus_bbox.jsonl -tp 8

# python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/06_vessel_nerve_lymph_det_seg.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/06_vessel_nerve_lymph_det_seg.jsonl -tp 8

# python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/07_cancer_in_vessel_nerve_lymph.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/07_cancer_in_vessel_nerve_lymph.jsonl -tp 8

python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/08_patch_subtyping.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/08_patch_subtyping.jsonl -tp 8

python zsxm_val/code/nips_infer.py --model-path zsxm_checkpoint/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/v0-20250416-131443/merged_checkpoint-9444 --question-file zsxm_dataset/nips/9_test/09_patch_grading.json --answers-file zsxm_val/results/nips/1_sft/qwen2.5-vl-32b_0415_VA-full-L-lora_2/09_patch_grading.jsonl -tp 8