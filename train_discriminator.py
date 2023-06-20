import os
import random, json
import numpy as np
import torch
from constants import *
from data_utils.discriminator_dataset import GSMPairwiseRankingDataset
from transformers import ElectraTokenizer, T5Tokenizer, AutoTokenizer
from reason_utils.electra_discriminator import ELECTRAEnergyDiscriminator
from reason_utils.t5_discriminator import T5EnergyDiscriminator
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from reason_utils.discriminator_trainer import DiscriminatorMaxMarginTrainer
from reason_utils.args import DiscriminatorDataArguments, DiscriminatorTrainingArguments, DiscriminatorModelArguments
from transformers.hf_argparser import HfArgumentParser
import transformers
import argparse
import dataclasses

## set logging level to info
transformers.logging.set_verbosity_info()

def test_tokenizer(tokenizer):
    test_string = '<< 5 + 6 = 11 >>'
    assert tokenizer.decode(tokenizer.encode(test_string, add_special_tokens=False)) == test_string, "Tokenizer is not working properly '{}' != '{}'".format(tokenizer.decode(tokenizer.encode(test_string, add_special_tokens=False)), test_string)


dataset_to_dataset_class = {
    "gsm8k": GSMPairwiseRankingDataset,
    "mathqa": GSMPairwiseRankingDataset,
    "multiarith": GSMPairwiseRankingDataset,
    "svamp": GSMPairwiseRankingDataset,
    "last_letter_concatenation": GSMPairwiseRankingDataset,
    "coin_flip": GSMPairwiseRankingDataset,
}

def main(data_args, train_args, model_args):

    if train_args.ckpt_dir is None:
        if 't5' in model_args.model:
            tokenizer = T5Tokenizer.from_pretrained(model_args.model)
        elif 'electra' in model_args.model:
            tokenizer = ElectraTokenizer.from_pretrained(model_args.model)
        if not train_args.fix_tokenizer:
            tokenizer.add_tokens(['<<', '>>'])
    else:
        if 't5' in model_args.model:
            tokenizer = T5Tokenizer.from_pretrained(train_args.ckpt_dir)
        elif 'electra' in model_args.model:
            tokenizer = ElectraTokenizer.from_pretrained(train_args.ckpt_dir)

    ## if cls and sep are not already added, add them
    if not tokenizer.cls_token:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if not tokenizer.sep_token:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    if not train_args.fix_tokenizer:
        test_tokenizer(tokenizer)

    ## load train and val examples 
    with open(data_args.trajectory_path) as f:
        samples = [json.loads(line) for line in f]

    ## split into train and val
    if data_args.dev_is_train:
        train_examples = samples
        val_examples = samples
    else:
        train_examples, val_examples = train_test_split(samples, test_size=0.15, random_state=11)

    if data_args.n_examples:
        train_examples = train_examples[:data_args.n_examples]
        val_examples = val_examples[:data_args.n_examples]
    
    print("Task: ", data_args.task)
    print("Loaded {} train examples and {} val examples".format(len(train_examples), len(val_examples)))
    if data_args.dev_is_train:
        print("Using train set as validation set")

    os.makedirs(train_args.output_dir, exist_ok=True)

    ### saving train and val examples to avoid re-creating the dataset everytime. 
    train_ds_path = os.path.join(train_args.output_dir, 'train_dataset.pt')
    val_ds_path = os.path.join(train_args.output_dir, 'val_dataset.pt')
    
    if os.path.exists(train_ds_path) and os.path.exists(val_ds_path):
        print("Loading train and val datasets")
        train_dataset = torch.load(train_ds_path, map_location=torch.device('cpu'))
        val_dataset = torch.load(val_ds_path, map_location=torch.device('cpu'))
        train_dataset.max_len = data_args.max_len
        val_dataset.max_len = data_args.max_len
    else:
        print("Creating train and val datasets")
        dataset_cls = dataset_to_dataset_class[data_args.task]
        train_dataset = dataset_cls(tokenizer=tokenizer, samples=train_examples, args=data_args)
        val_dataset = dataset_cls(tokenizer=tokenizer, samples=val_examples, args=data_args)
        #torch.save(train_dataset, train_ds_path)
        #torch.save(val_dataset, val_ds_path)


    print("Got a total of {} train examples and {} val examples".format(len(train_dataset), len(val_dataset)))
    if 'electra' in model_args.model:
        model = ELECTRAEnergyDiscriminator(model_name_or_path=model_args.model, args=model_args)
        model.model.resize_token_embeddings(len(tokenizer))
    elif 't5' in model_args.model:
        model = T5EnergyDiscriminator(model_name_or_path=model_args.model, args=model_args)

    else:
        raise ValueError("Model type not supported: {}".format(model_args.model))
    
    model.model.resize_token_embeddings(len(tokenizer))
    
    if train_args.ckpt_dir is not None:
        ckpt = torch.load(os.path.join(train_args.ckpt_dir, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model.load_state_dict(ckpt)

    trainer = DiscriminatorMaxMarginTrainer(model=model,
                                            tokenizer=tokenizer,
                                            train_dataset=train_dataset,
                                            eval_dataset=val_dataset,
                                            data_collator=train_dataset.collate_fn,
                                            args=train_args)
    trainer.train()

    ## save best model to output_dir/best_model
    outdir = os.path.join(train_args.output_dir, 'best_model')
    os.makedirs(outdir, exist_ok=True)
    trainer.save_model(os.path.join(train_args.output_dir, 'best_model'))
    
    ## save data_args 
    with open(os.path.join(outdir, 'data_args.json'), 'w') as f:
        json.dump(dataclasses.asdict(data_args), f, indent=4)

    ## save train_args
    with open(os.path.join(outdir, 'train_args.json'), 'w') as f:
        json.dump(dataclasses.asdict(train_args), f, indent=4)
    
    ## save model_args
    with open(os.path.join(outdir, 'model_args.json'), 'w') as f:
        json.dump(dataclasses.asdict(model_args), f, indent=4)
    
    

if __name__=='__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_file', type=str, default=None, required=False)
    args, _ = cli_parser.parse_known_args()

    parser = HfArgumentParser((DiscriminatorDataArguments, DiscriminatorTrainingArguments, DiscriminatorModelArguments))

    if args.config_file is None:
        hf_args = parser.parse_args_into_dataclasses()
    else:
        hf_args = parser.parse_json_file(json_file=args.config_file, allow_extra_keys=True)

    wandb.init(project="reaso-dec", entity="muhammad-khalifa")
    wandb.config.update(args)

    ## pretty print args
    #print('args', json.dumps(vars(args), indent=4, sort_keys=True))
    data_args, training_args, model_args = hf_args

    ## non-hf args
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    
    main(data_args, training_args, model_args)
