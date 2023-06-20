#!/bin/sh

TASK=$1

echo "seed = $seed"
python evaluate_reasoning.py \
			--model_name_or_path ckpts/lm/flan-t5-large_$TASK/best_model/ \
			--in_file data/$TASK/dev.jsonl \
			--task $TASK \
			--disc_path ckpts/discrim/ \
			--condition_lambda 0.0 --precondition_topk 20 --generation_type step-score \
			--step_sampling_method top_p --device2 cuda:0 --top_p .95 --sample_calc true \
			--generator_only false  --max_steps 6  --max_step_length 60 --step_delimiter '|' --temperature .8  --n_self_consistency 1 --seed $seed
			
