#!/bin/sh
set -e

TASK=$1

WANDB_MODE=disabled python sample_negative_solutions.py --in_file data/$TASK/train.jsonl \
				--model_name_or_path ../reasoning-decoding/ckpts/generator/google/flan-t5-large_gsm8k/best_model/ \
		        	--task $TASK  --temperature 1.3 \
				--model_tokenizer_path ../reasoning-decoding/ckpts/generator/google/flan-t5-large_gsm8k/ \
				--n_total_samples 80000 --top_k 50 \
				--out_dir data/discriminator_data/sampled_trajectories/gen.t5.large/${TASK}/ \
				--do_sample --seed 23 \
				--bf16 --batch_size 256  --max_length 256 --sample_calc

