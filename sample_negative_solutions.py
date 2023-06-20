import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from constants import *
import json
from data_utils.utils import prepare_icl_input, create_demos, is_correct, is_correct_program, ANS_RE, LLC_ANS_RE, strip_computations, extract_answer_llc
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from grace.args import TASKS

def main(args):
    if args.model_tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer_path, padding_side='right' if 't5' in args.model_name_or_path else 'left')

    if 't5' in args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, 
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(args.device)
    elif 'llama' in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float32, load_in_8bit=True, 
                device_map="auto")
        
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    demos = None

    input_examples = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            d = json.loads(line)
            input_examples.append(d)
            
    if args.n_examples is not None:
        input_examples = input_examples[:args.n_examples]
        
    if args.icl:
        ## assert demos are not in the inputs
        demos_examples = []
        ## load demos 
        print("Using In-context learning with {} demos".format(args.n_demos))
        demos_path = args.demos_file
        with open(demos_path, "r") as f:
            for line in f:
                demo = json.loads(line)
                ## add eos token to the end of the demo
                demos_examples.append(demo)
                assert demo['question'] not in [a['question'] for a in input_examples], "The demo {} is in the eval examples!".format(demo['question'])
        
    ## process demos
    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
    
    ## generating trajectories
    print("Generating trajectories...")
    question_to_traj = defaultdict(dict)
    bsz = args.batch_size

    n_correct = 0
    n_total = 0
    n_unique = 0

    if not args.icl: # not few-shot 
        if args.task in ["gsm8k", "svamp", "multiarith",                "last_letter_concatenation", "tso"]:
            STEP_DELIMITER = "|"
        elif args.task in ["asdiv", "mathqa"]:
            STEP_DELIMITER = ";"
        else:
            raise NotImplementedError("Task {} not supported".format(args.task))
    else:
        STEP_DELIMITER = "." # for icl, we use . as the delimiter by default
    
    if args.step_delimiter is not None:
        STEP_DELIMITER = args.step_delimiter

    assert STEP_DELIMITER in ['.', '|', ';'], "Step delimiter {} not supported".format(STEP_DELIMITER)
    
    if args.task in ["gsm8k", "svamp", "multiarith", "mathqa", "asdiv"]:
        ans_re = ANS_RE
    elif args.task in ["last_letter_concatenation", "coin_flip", "tso"]:
        ans_re = LLC_ANS_RE
    else:
        raise NotImplementedError("Task {} not supported".format(args.task))

    while n_unique < args.n_total_samples:
        ## pick random demos to diversify the sampled trajectories
        for i in tqdm(range(0, len(input_examples), bsz), disable=False):
            if args.icl:
                print("Sampling new demos...")
                demos = random.sample(demos_examples, args.n_demos)
                demos = create_demos(demos, step_delimiter=STEP_DELIMITER)
            
            qns = [a['question'] for a in input_examples[i:i+bsz]]
            qns_prepared = [prepare_icl_input(qn, demos=demos, instruction=args.instruction) for qn in qns]
            batch = tokenizer(qns_prepared, 
                            padding=True,
                            return_tensors="pt")

            batch = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            
            generated = model.generate(**batch, max_new_tokens=args.max_length,
                                                num_return_sequences=1,
                                                do_sample=True, 
                                                top_k=args.top_k,
                                                temperature=args.temperature,
                                                top_p=args.top_p,
                                                sample_calc=args.sample_calc,
                                                tokenizer=tokenizer, 
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id,
                                                )

            if 't5' in args.model_name_or_path:
                generated = tokenizer.batch_decode(generated, skip_special_tokens=False)
                gens = [g.replace('<unk> ','<<').replace('<pad>','').strip() for g in generated] # HACK since << is not a token in the default T5 tokenizer
                generated = [g.split('</s>')[0] for g in gens]
                
            elif 'llama' in args.model_name_or_path:
                ## get new tokens only 
                gens = []
                for g in generated:
                    new_tokens = g[batch['input_ids'].shape[1]:]
                    gens.append(new_tokens)
                gens = tokenizer.batch_decode(gens, skip_special_tokens=True)
                generated = []
                for g in gens:         
                    match = ans_re.search(g)
                    if match:
                        ans = match.group(0).strip()
                        g = g[:g.find(ans)+len(ans)] 

                    generated.append(g)

            assert len(generated) == len(qns)

            gt_ans = [ex["answer"] for ex in input_examples[i:i+bsz]]
            sol_labels = [is_correct(p, gt, task=args.task) for gt, p in zip(gt_ans, generated)]

            n_correct += sum(sol_labels)
            n_total += len(sol_labels)

            ## because some ground-truth are not well seopareted (those sampled from LLMs), we need to do some post-processing
            for i, g in enumerate(gt_ans):
                g = g.replace("  \n", " \n").replace(" \n", "\n").replace(".\n",
                        "\n").replace("\n", STEP_DELIMITER)
                if STEP_DELIMITER is not None and STEP_DELIMITER != '.': # to make sure gt is well separated
                    gsents = sent_tokenize(g)
                    ## remove periods
                    gsents = [s.rstrip('.') for s in gsents]
                    g = STEP_DELIMITER.join(gsents)
                gt_ans[i] = g

            for qn, traj, lbl, gt_a in zip(qns, generated, sol_labels, gt_ans):
                if not qn in question_to_traj:
                    question_to_traj[qn]['trajectories'] = []
                    question_to_traj[qn]['is_correct'] = []
                    question_to_traj[qn]['gt_sol'] = gt_a
                
                if traj not in question_to_traj[qn]['trajectories']: ## only add unique trajectories
                    question_to_traj[qn]['trajectories'].append(traj)
                    question_to_traj[qn]['is_correct'].append(lbl)

                    if not lbl:
                        n_unique += 1 # only count unique incorrect trajectories
                
        print("Sampled a total of {} INCORRECT trajectories".format(n_unique))
        print("So far, got %.2f%% correct trajectories" % (n_correct * 100.0 / n_total))

        ## round unique to nearest 1000
        n_unique_rnd = int(round(n_unique, -3))
        if args.out_dir and n_unique_rnd % 5000 == 0:
            print("Caching trajectories to disk...")
            out_file = os.path.join(args.out_dir, 'trajectories_seed{}_{}.jsonl'.format(args.seed, n_unique_rnd))
            with open(out_file, 'w') as wf:
                for qn, traj in question_to_traj.items():
                    wf.write(json.dumps({'question': qn, 'trajectories': traj['trajectories'], 'is_correct': traj['is_correct'], 'gt_sol': traj['gt_sol']}) + '\n')

            ## save sampling args 
            args_file = os.path.join(args.out_dir, 'args.json')
            with open(args_file, 'w') as wf:
                json.dump(vars(args), wf)
      
    if args.out_dir: 
        out_file = os.path.join(args.out_dir, 'trajectories_seed{}.jsonl'.format(args.seed))
        with open(out_file, 'w') as wf:
            for qn, traj in question_to_traj.items():
                wf.write(json.dumps({'question': qn, 'trajectories': traj['trajectories'], 'is_correct': traj['is_correct'], 'gt_sol': traj['gt_sol']}) + '\n')

    
    print("Finished generating trajectories.")
    print("Got {} unique trajectories".format(n_unique))
    print("Got %.2f%% correct trajectories" % (n_correct * 100.0 / n_total))
    print("Diverse trajectories percentage= %.2f%%" % (n_unique * 100.0 / n_total))


if __name__=='__main__':
    parser = ArgumentParser()
    # DATA
    parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-large')
    parser.add_argument('--model_tokenizer_path', type=str, default=None)
    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing text to run pred on')
    parser.add_argument('--task', type=str, default='gsm8k', choices=TASKS)
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
    parser.add_argument('--top_k', type=int, default=None, help='top k for sampling')
    parser.add_argument('--top_p', type=float, default=None, help='top p for sampling')
    parser.add_argument('--do_sample', action='store_true', default=True)
    parser.add_argument('--n_samples_per_example', type=int, default=100, help='number of samples to generate')
    parser.add_argument('--n_total_samples', type=int, default=100000, help='max total samples to generate')
    parser.add_argument('--max_length', type=int, default=200, help='max length')
    parser.add_argument('--min_length', type=int, default=200, help='min length')
    parser.add_argument('--num_beams', type=int, default=1, help='beam size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    ## ICL and sampling
    parser.add_argument('--icl', action='store_true', default=False, help='use icl')
    parser.add_argument('--n_demos', type=int, default=2, help='number of demonstrations')
    parser.add_argument('--demos_file', type=str, default=None, help='file containing demonstrations')
    parser.add_argument('--instruction', type=str, default='Solve the following math problems.', help='instruction to prepend to input')
    parser.add_argument('--out_dir', type=str, default=None, help='directory to save trajectories')
    parser.add_argument('--sample_calc', action='store_true', default=False, help='sample using calculator')
    parser.add_argument('--step_delimiter', type=str, default=None)
    ## other
    parser.add_argument('--n_examples', type=int, default=None, help='number of examples to run')
    ### model precision
    parser.add_argument('--fp16', action='store_true', default=False, help='use fp16')
    parser.add_argument('--bf16', action='store_true', default=False, help='use bf16')

    args = parser.parse_args()

    ## set seed based on time if not specified
    if args.seed == -1:
        args.seed = int(time.time())
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
