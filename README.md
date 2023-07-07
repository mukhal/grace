# GRACE Decoding
Discriminator-guided Multi-step Reasoning with Language Models - [[**Preprint**]](https://github.com/mukhal/grace), [[**Website**]](https://mukhal.github.io/grace/)

![image](https://github.com/mukhal/grace-decoding/assets/5109053/cdb93474-1613-47d8-9bf4-be2ae3086979)

# Instructions 

## Setup 
The experiments in the paper were run using `torch==1.13.0`. You also need to install the transformers version included in this repo since it was modified to support *calculator* sampling. 
```
cd transformers/
pip install -e .
```



## Discriminator Training
To show you how to train the discriminator, we will use GSM8K to illustrate. 
### Step 1: Negative Sampling 
```
WANDB_MODE=disabled python sample_negative_solutions.py --in_file data/$TASK/train.jsonl \
                                --model_name_or_path path-to-lm \
                                --task gsm8k  --temperature 1.3 \
                                --model_tokenizer_path .path-to-lm \
                                --n_total_samples 80000 --top_k 50 \
                                --out_dir path-to-sampled-solutions \
                                --do_sample --seed 23 \
                                --bf16 --batch_size 256  --max_length 256 --sample_calc
```
All parameters are self-explanatory, but `--sample_calc` means we will use calculator sampling. That is whenever an operation such as `<< 4 + 5=9 >>` is generated, we will invoke a calculator module to compute the result. 

### Steps 2 and 3: Alignment and Discriminator Training
Now we want to train a FLAN-T5 encoder as a discriminator over the sampled solutions. 
```
accelerate launch  --mixed_precision=bf16  --num_processes=$GPUS_PER_NODE train_discriminator.py  --task gsm8k \
                        --trajectory_path path-to-sampled-solutions/trajectories.jsonl \
                        --model google/flan-t5-large \
                        --output_dir ckpts/discrim/flan_t5_large_gsm8k/ \
                        --evaluation_strategy steps --save_steps 2000 --eval_steps  2000 --save_total_limit 2 \
                        --prediction_loss_only True \
                        --max_len 512  --margin 1.0 --logging_steps 50 --warmup_ratio 0.06 \
                        --lr_scheduler_type cosine_with_restarts --weight_decay 0.01 \
                        --per_device_train_batch_size 8 --num_train_epochs 5 --learning_rate 1e-4  \
                        --metric_for_best_model 'loss' --greater_is_better False --load_best_model_at_end True \
                        --bf16 --gradient_accumulation_steps 2 --pooling "max" --report_to 'none' \
                        --step_aligner_model roscoe --max_alignment_cost 3.0
```
All parameters are self-explanatory too, except for: 
* `--step_aligner_model`: the model used for alignment (`roscoe`, `simcse`, or `openai` embeddings).
* `--max_alignment_cost`: The maximum alignment cost between a sampled solution and the reference solution. If the alignment cost is not above this value, the sampled solution is not used to create examples for the discriminator



## Stepwise Guided Decoding
Now we run the guided decoding using the trained discriminator. 
```
WANDB_MODE=disabled python run_grace.py \
                        --model_name_or_path path-to-lm/ \
                        --in_file data/gsm8k/dev.jsonl \
                        --task gsm8k \
                        --disc_path ckpts/discrim/flan_t5_large_gsm8k/ \
                        --beta 0.1 --n_candidate_steps 20 --generation_type step-score \
                        --step_sampling_method top_p --device2 cuda:0 --top_p .95 --sample_calc true \
                        --max_steps 6  --max_step_length 60 --step_delimiter '|' --temperature .8  --n_self_consistency 1 --seed 42
```
relevant arguments are:
* `--generation_type`: how we guide decoding. `step-score` is the method described in the paper.
* `--step_delimiter`: the step delimiter token used (used to stop generation at the end of a given steps when sampling candidate steps). We use `|` for gsm8k, SVAMP, and MultiArith and `;` for MathQA.
* `--n_self_consistency`: the number of samples to use for self-consistency with grace. If set to one, then no majority voting is applied.
* `--max_steps`: Maximum steps per sampled solution.
* `--step_sampling_method`: how we sample candidate steps
* `--n_candidate_steps`: number of candidate steps to sample and score.


## Trained Models
We will upload the fine-tuned models and discriminators used in the paper soon. 

## Coming Soon
A multitask-trained discriminator on several reasoning tasks! 

## Citation
If you use this code, please consider citing out paper
```
@article{khalifa2023discriminator,
  title={Discriminator-Guided Multi-step Reasoning with Language Models},
  author={Khalifa, Muhammad and Logeswaran, Lajanugen and Lee, Moontae and Lee, Honglak and Wang, Lu},
  journal={arXiv preprint arXiv:2305.14934},
  year={2023}
}
```
