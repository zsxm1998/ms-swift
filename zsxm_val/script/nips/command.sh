bash zsxm_val/script/nips/eval/01_thumbnail_seg.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/07_patch_subtyping.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484

bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think; bash zsxm_val/script/nips/eval/02_thumbnail_choice_additional.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/02_thumbnail_choice_additional.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think

bash zsxm_val/script/nips/eval/03_nucleus_det_no_class.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think --func; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --func

bash zsxm_val/script/nips/eval/04_nucleus_det_with_class.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/02_thumbnail_choice_additional.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think --func; bash zsxm_val/script/nips/eval/02_thumbnail_choice_additional.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --func

bash zsxm_val/script/nips/eval/05_vessel_nerve_lymph_det_seg.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/08_patch_grading.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484

bash zsxm_val/script/nips/eval/06_cancer_in_vessel_nerve_lymph.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/09_classify_nucleus_bbox.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484; bash zsxm_val/script/nips/eval/09_classify_nucleus_bbox.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think

bash zsxm_val/script/nips/eval/07_patch_subtyping.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think

bash zsxm_val/script/nips/eval/08_patch_grading.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think




bash zsxm_val/script/nips/eval/02_thumbnail_choice_additional.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --func --batch-size 2; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --func --batch-size 2

bash zsxm_val/script/nips/eval/02_thumbnail_choice_additional.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think --func --batch-size 2; bash zsxm_val/script/nips/eval/02_thumbnail_choice.sh zsxm_checkpoint/nips/1_sft/omnipt-qwen2-vl-7b_0510_full_VAL_2/v0-20250609-155108/checkpoint-9484 --think --func --batch-size 2