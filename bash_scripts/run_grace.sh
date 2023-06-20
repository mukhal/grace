#!/bin/sh

TASK=$1

WANDB_MODE=disabled python run_grace.py \
			--model_name_or_path ../reasoning-decoding/ckpts/generator/google/flan-t5-large_$TASK/best_model/ \
			--in_file data/$TASK/test.jsonl \
			--task $TASK \
			--disc_path ../reasoning-decoding/ckpts/discrim/contrastive_models/flan_t5_large_gsm8k_max_margin_invalid_prefixes/best_model/ \
			--beta 0.1 --n_candidate_steps 20 --generation_type step-score \
			--step_sampling_method random --device2 cuda:0 --top_p .95 --sample_calc true \
			--generator_only false  --max_steps 6  --max_step_length 60 --step_delimiter '|' --temperature .8  --n_self_consistency 1 --seed 42
			
