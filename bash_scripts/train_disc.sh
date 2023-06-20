#!/bin/sh
set -e
export TRANSFORMERS_OFFLINE=0
export GPUS_PER_NODE=2
export WANDB_MODE=disabled

TASK=$1

accelerate launch  --mixed_precision=bf16  --num_processes=$GPUS_PER_NODE train_discriminator.py  --task $TASK \
                        --trajectory_path data/discriminator_data/sampled_trajectories/gen.t5.large/$TASK/trajectories.jsonl \
                        --model google/flan-t5-large \
                        --output_dir ckpts/discrim/flan_t5_large_$TASK/ \
                        --evaluation_strategy steps --save_steps 2000 --eval_steps  2000 --save_total_limit 2 \
            		--prediction_loss_only True \
		        --max_len 512  --margin 1.0 --logging_steps 50 --warmup_ratio 0.06 \
	 		--lr_scheduler_type cosine_with_restarts --weight_decay 0.01 \
			--per_device_train_batch_size 8 --num_train_epochs 5 --learning_rate 1e-4  \
			--metric_for_best_model 'loss' --greater_is_better False --load_best_model_at_end True \
			--bf16 --gradient_accumulation_steps 2 --pooling "max" --invalid_prefix_prob 0.1 --report_to 'none' \
                        --step_aligner_model roscoe --skip_alignment
                    

