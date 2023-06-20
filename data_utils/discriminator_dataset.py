import json, os, math
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import torch as th
import random
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from alignment_utils import StepAligner
from constants import ANS_IDENTIFIER
from torch.nn.utils.rnn import pad_sequence
from data_utils.utils import timeout
import torch


TASK_TO_STEP_DELIMITER = {
    "gsm8k": "|",
    "coin_flip": "|",
    "mathqa": ";",
    "multiarith": "|",
    "svamp": "|",
    "asdiv": ";",
    "last_letter_concatenation": "|",
    "tso": "|",
}


class GSMPrefixDiscriminatorDataset(th.utils.data.Dataset):
    '''
    Discriminator dataset class that creates negative examples by perturbatng valid trajectories
    '''
    def __init__(self, tokenizer, examples, loss_on_prefix=True, args=None): 
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.max_len = getattr(args, "max_len", 150)
        self.tokenizer = tokenizer
        # filter out examples that are too long
        #self.examples = [ex for ex in self.examples if len(tokenizer.tokenize(ex["question"])) <= self.max_len and len(tokenizer.tokenize(ex["answer"])) <= self.max_len]
        
        self.corrupt_prob = getattr(args, "corrupt_prob", 0.5)
        self.step_delimiter = getattr(args, "step_delimiter", ". ")
        #self.tokenizer.add_special_tokens({"additional_special_tokens": [self.step_delimiter]})

        ## clear unnnecessary whitespace
        for ex in self.examples:
            ##
            ex["answer"] = ' '.join(ex["answer"].split())
            ex["question"] = ' '.join(ex["question"].split())
        
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        #qn_tokens = self.qns["input_ids"][idx]
        #ans_tokens = self.ans["input_ids"][idx]

        ## fix random seed so that the same corruption is applied to the question and answer
        #random.seed(idx)
        qn_text = self.examples[idx]["question"]
        ans_text = self.examples[idx]["answer"]
        qn_tokens, ans_tokens, labels = self.maybe_corrupt_example(qn_text, ans_text)

        ## truncate if too long
        qn_tokens = qn_tokens[:self.max_len]
        ans_tokens = ans_tokens[:self.max_len]
        labels = labels[:self.max_len]

        assert len(qn_tokens) <= self.max_len, f"question too long: {qn_text}"
        assert  len(ans_tokens) <= self.max_len, f"answer too long: {ans_text}"
        assert len(ans_tokens) == len(labels), "lengths don't match!"
        
        # pad to self.max_len 
        qn_tokens = qn_tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(qn_tokens))
        ans_tokens = ans_tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(ans_tokens))

        # attention masks 
        qn_mask = [1] * len(qn_tokens) + [0] * (self.max_len - len(qn_tokens))
        ans_mask = [1] * len(ans_tokens) + [0] * (self.max_len - len(ans_tokens))
        labels = labels + [0] * (self.max_len - len(labels))
        
        enc_input_tokens = th.tensor(qn_tokens)
        dec_input_tokens = th.tensor(ans_tokens)
        enc_input_mask = th.tensor(qn_mask)
        dec_input_mask = th.tensor(ans_mask)
        labels = th.tensor(labels)
        
        return dict(enc_input_ids=enc_input_tokens, dec_input_ids=dec_input_tokens, 
                                enc_attn_mask=enc_input_mask, 
                                dec_attn_mask=dec_input_mask, labels=labels)

    def get_tokens_and_labels(self, text_pre, text_post):
        """
        Takes two versions of text before and after corruption.
        Returns the tokenized text and the labels for the discriminator.
        Labels are all ones until the first corruption, then all zeros.
        """
        
        ## tokenize
        tokens_pre = self.tokenizer(text_pre, padding=False)["input_ids"]
        tokens_post = self.tokenizer(text_post, padding=False)["input_ids"]
        
        ## get the first index where the two texts differ
        for i in range(min(len(tokens_pre), len(tokens_post))):
            if tokens_pre[i] != tokens_post[i]:
                break
        
        if i == min(len(tokens_pre), len(tokens_post)) - 1:
            ## if the two texts are the same, then just return the text as is
            return tokens_post, [1] * len(tokens_post)
        
        ## label all tokens before the first difference as 1, and all after as 0
        labels = [1] * i + [0] * (len(tokens_post) - i)
        return tokens_post,  labels
    
    def replace_substrings(self, text, token, replace, num_replacements=1, replace_all=False):
        # from https://stackoverflow.com/questions/10060149/python-how-to-replace-n-random-string-occurrences-in-text
        num_tokens = text.count(token)
        if replace_all:
            num_replacements = num_tokens
            points = [0] + list(range(1, num_tokens)) + [num_tokens + 1]
        else:
            points = [0] + sorted(random.sample(range(1, num_tokens + 1), num_replacements)) + [num_tokens + 1]
        
        return replace.join(token.join(text.split(token)[i:j]) for i, j in zip(points, points[1:]))

    def corrupt_numbers(self, text):
        """
        Swaps a number with another random number to create a corrupt question. 
        """
        org_text = text
        numbers = list(set(re.findall(r"\d+", text)))
        if len(numbers) < 2: # can't corrupt if there's only one number
            return text
        # randomly choose one number
        number = random.choice(numbers)
        # randomly choose a number to replace it with
        new_number = random.choice([n for n in numbers if n != number])
        # swap occurrences of the two numbers
        text = text.replace(new_number, '[MASK]')
        text = text.replace(number, new_number)
        text = text.replace('[MASK]', number)
        
        return text
    
    
    def shuffle_steps(self, text):
        """
        swap two random steps 
        """
        org_text = text
        steps = sent_tokenize(text) # a step is just a sentence
        if len(steps) < 3: # can't corrupt if there's only one step other than the final answer
            return text
        # randomly choose two steps
        possible_idxs = list(range(len(steps) - 1)) # don't shuffle final answer
        step1_idx= random.choices(possible_idxs, k=1)[0] 
        # steps with higher index more likely to avoid having all 0 labels too often.
        step2_idx = random.choices([i for i in range(len(possible_idxs)) if i != step1_idx], k=1)[0]
        steps[step1_idx], steps[step2_idx] = steps[step2_idx], steps[step1_idx]
        text = ' '.join(steps)
        return text

    def drop_step(self, text):
        """
        drop a random step
        """
        org_text = text
        steps = sent_tokenize(text)
        if len(steps) < 2:
            return text
        # randomly choose a step
        possible_idxs = list(range(len(steps) - 1)) # don't drop final answer
        step_idx = random.choices(possible_idxs, k=1)[0]
        steps.pop(step_idx)
        text = ' '.join(steps)
        return text

    def corrupt_operators(self, text):
        """
        replaces a random operator with another
        """
        # find +-/* surrounded by numbers
        org_text = text
        operators = re.findall(r"\d[\+\-\*\/]\d", text)
        operators = list(set(s[1] for s in operators if s[1] in ['+', '-', '*', '/']))
        if len(operators) <= 1:
            return text 
        # randomly choose two operators
        op1= random.choices(operators, k=1)[0]
        op2 = random.choice([op for op in ['+', '-', '*', '/'] if op != op1])
        text = re.sub(r"(\d)\%s(\d)" % op1, r"\1{}\2".format(op2), text)

        return text
    
    def corrupt_final_answer(self, text):
        """
        swap the final answer with a random number 
        """
        org_text = text
        final_answer = re.findall(r"\d+", text)[-1]
        numbers = list(set(re.findall(r"\d+", text)))
        if len(numbers) < 2:
            return text
        # randomly choose a number to replace it with
        new_number = random.choice([n for n in numbers if n != final_answer])
        # swap occurrences of the two numbers
        text = text.replace('#### ' + final_answer, '#### ' + new_number)
        assert text != org_text
        return text

    def maybe_corrupt_example(self, qn_text, ans_text):
        """
        minor corruption of the question
        """
        ## flip a coin 
        if random.random() < 1 - self.corrupt_prob: #1 - self.corrupt_prob: # don't corrupt
            tokenized_qn = self.tokenizer(qn_text, padding=False)["input_ids"]
            tokenized_ans = self.tokenizer(ans_text, padding=False)["input_ids"]
            labels = [1] * len(tokenized_ans)
            return (tokenized_qn, tokenized_ans, labels)
        
        p_number_corrupt = 0.4
        p_corrupt_steps = 0.15
        p_operator_corrupt = 0.4
        p_final_answer_corrupt = 0.05
        qn_org, ans_org = qn_text, ans_text

        q_corrupt = False
        if random.random() < 0.0: # turn off for now
            q_corrupt = True
            if random.random() < p_number_corrupt:
                qn_text = self.corrupt_numbers(qn_text)
        else: # corrupt answer -- one corruption at a time
            if random.random() < p_number_corrupt:
                ans_text = self.corrupt_numbers(ans_text)
            elif random.random() < p_corrupt_steps:
                if random.random() < 0.5:
                    ans_text = self.shuffle_steps(ans_text)
                else:
                    ans_text = self.drop_step(ans_text)
            elif random.random() < p_operator_corrupt:
                ans_text = self.corrupt_operators(ans_text)
            elif random.random() < p_final_answer_corrupt:
                ans_text = self.corrupt_final_answer(ans_text)

        if q_corrupt: # NOTE: even if the question is corrupted, it doesn't mean all the answer is wrong
            tokenized_qn = self.tokenizer(qn_text, padding=False)["input_ids"]
            tokenized_ans = self.tokenizer(ans_org, padding=False)["input_ids"]
            labels = [0] * len(tokenized_ans)
            return (tokenized_qn, tokenized_ans, labels)
        else:
            ans_tokens, labels = self.get_tokens_and_labels(ans_org, ans_text)
            return (self.tokenizer(qn_text, padding=False)["input_ids"],
                    ans_tokens, labels)
            
class GSMPrefixModelSamplesDataset(th.utils.data.Dataset):
    '''
    Discriminator dataset based on samples from the generator
    '''

    def __init__(self, samples, tokenizer, args=None):
        '''
        Args:
            samples: list of dicts with keys: question and values: trajectories, is_correct, gt_sol
            tokenizer: huggingface tokenizer
            max_len: max length of input sequence
        '''
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.alignment = getattr(args, 'alignment')
        self.labeling = getattr(args, 'labeling')
        self.step_delimiter = TASK_TO_STEP_DELIMITER[args.task]
        self.only_first_error = getattr(args, 'only_first_error') # labels will only consist of 1 1 1 1 followed by a single 0 to avoid unnecessary optimization of invalid suffixes
        self.args = args
        self.stitch = getattr(args, 'stitch')
        self.model_style = 'enc-dec' if 't5' in args.model else 'enc'
    
        if not self.stitch:
            assert not self.only_first_error, "--only_first_error can only be used with stitching"

        if self.alignment == 'step':
            print("Building stepwise dataset...")
            self.examples = self.step_align_and_get_labels(self.samples, args)
        elif self.alignment == 'token':
            print("Building tokenwise dataset...")
            self.examples = self.token_align_and_get_labels(self.samples, args)
        else:
            raise NotImplementedError

        ## compute percentage of positive labels
        n_pos, n_tot = 0, 0
        for ex in self.examples:
            n_pos += sum([a for a in ex[2] if a == 1])
            n_tot += len([a for a in ex[2] if a >= 0])
        print("Percentage of positive labels: %.2f" % (n_pos / n_tot))

    def _get_tokens_labels_by_stitching_tokens(self, gt_sol, traj):
        ## returns a list of tokens and labels by doing trajectory stitching of valid prefixes from the ground truth with invalid suffixes from the sampled trajectory.
        ## tokenize both the ground truth and the sampled trajectory
        gt_tokens = self.tokenizer(gt_sol, padding=False, add_special_tokens=False)["input_ids"]
        traj_tokens = self.tokenizer(traj, padding=False, add_special_tokens=False)["input_ids"]

        ## find the longest valid prefix of the sampled trajectory that matches the ground truth
        min_len = min(len(gt_tokens), len(traj_tokens))
        ## start from the first mistake the model made and stitch the valid prefix of the ground truth with the sampled trajectory suffix
        for i in range(min_len):
            if gt_tokens[i] != traj_tokens[i]:
                ## a possible error here. Stitch the valid prefix of the ground truth with the sampled trajectory suffix
                #tokens_ = gt_tokens[:i] + self.tokenizer.encode(" ***> ", #add_special_tokens=False) +  traj_tokens[i:]
                if self.only_first_error:
                    tokens = gt_tokens[:i] +  [traj_tokens[i]]
                    labels = [1] * len(gt_tokens[:i]) + [0]
                else:
                    tokens = gt_tokens[:i] +  traj_tokens[i:]
                    labels = [1] * len(gt_tokens[:i]) + [0] * len(traj_tokens[i:])
                
                # TODO: do stitching at all errors
                return tokens, labels
            
    def _get_tokens_labels_no_stitching(self, gt_sol, traj):
        gt_tokens = self.tokenizer(gt_sol, padding=False, add_special_tokens=False)["input_ids"]
        traj_tokens = self.tokenizer(traj, padding=False, add_special_tokens=False)["input_ids"]

        min_len = min(len(gt_tokens), len(traj_tokens))
        # labels is 1 if the token is correct, 0 otherwise
        labels = [1 if gt_tokens[i] == traj_tokens[i] else 0 for i in range(min_len)]
        traj_tokens = traj_tokens[:min_len]

        return traj_tokens, labels

    def token_align_and_get_labels(self, samples, args):
        '''
        Performs tokenwise aligmnment between sampled trajectories and ground truth ones
        Returns a list of tuples (tokenized question, tokenized answer, tokenwise labels)

        Args:
            samples: list of dicts with keys: question and values: trajectories, is_correct, gt_sol

        Questions to answer:
        Q: How should we treat trajectories with correct answers but different intermediate steps from the ground truth?
        A: We should treat them as incorrect. We can do this by aligning the tokens of the ground truth and the sampled trajectory (at least at first).

        Q: If the labels are 1, 1, 1, 0 0 0 0 0 0. How do we teach the model to avoid mistakes done after the first mitakes? Do we need trajectory stitching?
        A: trajectory stitching seems to be the only way to do this. We can do it by using the  ground truth as the prefix and the sampled trajectory as the suffix.

        '''
        examples = []
        for sample in tqdm(samples):
            question = sample['question']
            if not question.startswith('Q: '):
                question = 'Q: ' + question
            
            gt_sol = sample['gt_sol']
            if not gt_sol.startswith('A: '):
                gt_sol = 'A: ' + gt_sol
            
            is_correct = sample['is_correct']
            question_tokens = self.tokenizer(question, padding=False, add_special_tokens=False)["input_ids"]

            gt_sol = gt_sol.replace('\n', self.step_delimiter)
            gt_sol = ' '.join(gt_sol.split()) # remove extra spaces

            ## add gold trajectory
            gt_sol_tokens = self.tokenizer(gt_sol, padding=False, add_special_tokens=False)["input_ids"]
            labels = [1] * len(gt_sol_tokens)
            examples.append((question_tokens, gt_sol_tokens, labels))

            for traj, is_cor in zip(sample['trajectories'], is_correct):
                traj = traj.replace('\n', self.step_delimiter)
                traj = ' '.join(traj.split())
                if not traj.startswith('A: '):
                    traj = 'A: ' + traj
            
                if is_cor and traj != gt_sol: # alternatie correct path
                    traj_tokens = self.tokenizer(traj, padding=False)["input_ids"]
                    labels = [1] * len(traj_tokens)
                    examples.append((question_tokens, traj_tokens, labels))
                    continue # only consider incorrect trajectories
                
                ## invalid trajectory
                # stitch and get labels
                if self.stitch:
                    traj_tokens, labels = self._get_tokens_labels_by_stitching_tokens(gt_sol=gt_sol,
                    traj=traj)
                    examples.append((question_tokens, traj_tokens, labels))
                else:
                    traj_tokens, labels = self._get_tokens_labels_no_stitching(gt_sol=gt_sol,
                    traj=traj)
                    examples.append((question_tokens, traj_tokens, labels))
        
        return examples
         
    def step_align_and_get_labels(self, samples, args):
        '''
        Performs stepwise alignment between sampled trajectories and ground truth ones
        Returns a list of tuples (tokenized question, tokenized answer, stepwise labels)
        '''
        examples = []
        n_skipped = 0
        for sample in tqdm(samples):
            #print("*************************************************************")
            #print("Question: ", sample['question'])
            question = sample['question']
            if not question.startswith('Q: '):
                question = 'Q: ' + question
            
            gt_sol = sample['gt_sol'].replace('.\n', '\n').replace('\n', self.step_delimiter).strip()
            gt_sol = gt_sol.split(ANS_IDENTIFIER)[0]
            gt_sol = "A: " + gt_sol
            is_correct = sample['is_correct']
            
            question_tokens = self.tokenizer(question, padding=False, add_special_tokens=False)["input_ids"]

            if gt_sol.startswith('A: '):
                gt_sol = gt_sol[3:] # remove the 'A: ' prefix before splitting to steps
                gt_sol = ' '.join(gt_sol.split())

            if self.step_delimiter == '.':
                gt_steps = sent_tokenize(gt_sol) #.split(self.step_delimiter)
            elif self.step_delimiter == ' |':
                gt_steps = [s.strip() for s in gt_sol.split(self.step_delimiter) if s.strip()]   
                #gt_steps = [step + self.step_delimiter if not step.endswith(self.step_delimiter) else step for step in gt_sol.split(self.step_delimiter) if step.strip()]    
 
            else:
                raise NotImplementedError("Only . is supported as step delimiter")
            
            gt_steps = [step + self.step_delimiter if not step.endswith(self.step_delimiter) else step for step in gt_steps]
            gt_steps_a = [s for s in gt_steps]
            gt_steps_a[0] = "A: " + gt_steps_a[0]
            
            ## add full gold trajectory to examples if NOT stitching
            ## because we already add the gold trajectory steps if stitching
            if not self.stitch:
                gt_token_steps, gt_token_labels = self._get_token_labels_from_step_labels(gt_steps_a, [1] * len(gt_steps_a))
                examples.append((question_tokens, gt_token_steps, gt_token_labels))

            for i, traj in enumerate(sample['trajectories']):
                traj = traj.replace('A:', '').strip()
                traj = traj.replace('.\n', '\n').replace('\n', self.step_delimiter)
                traj = ' '.join(traj.split())

                ## if stitching, only consider incorrect trajectories
                if self.stitch and is_correct[i]:
                    #print("Skipping correct trajectory: ")
                    continue
                
                if ANS_IDENTIFIER in traj:
                    traj = traj.split(ANS_IDENTIFIER)[0]

                if self.step_delimiter == '.':
                    traj_steps = sent_tokenize(traj) #.split(self.step_delimiter)
                elif self.step_delimiter == ' |':
                    traj_steps = [s.strip() for s in traj.split(self.step_delimiter) if s.strip()]
                else:
                    raise NotImplementedError("Only . is supported as step delimiter for now.")

                #if abs(len(traj_steps) - len(gt_steps)) > 2 or len(traj_steps) < 2: ## more than two missing/extra step
                #    continue
                if len(traj_steps) < 2: ## more than two missing/extra step
                    continue

                ## make sure every traj step end with step delimiter
                traj_steps = [step + self.step_delimiter if not step.endswith(self.step_delimiter) else step for step in traj_steps]
                
                assert not traj_steps[0].startswith("A: ") and not gt_steps[0].startswith("A: ") # make sure the first step does not start with 'A: ' to not confuse the alignment algorithm

                aligned_traj, aligned_gt, cost = compute_alignment_from_trajectories(traj_steps, gt_steps, delimiter=self.step_delimiter)

                if cost > self.args.max_alignment_cost: ## roughly three gaps
                    n_skipped += 1
                    #print("skipped {} trajectories with high alignment cost.".format(n_skipped))
                    continue
                
                if not self.stitch:
                    steps, labels = self._get_steps_labels(aligned_traj, aligned_gt)
                    steps[0] = "A: " + steps[0] # add the 'A: ' prefix to the first step
                    token_steps, token_labels = self._get_token_labels_from_step_labels(steps, labels)
                    examples.append((question_tokens, token_steps, token_labels))
                else:
                    for steps, labels in self._get_steps_labels_by_stitching_steps(aligned_traj, aligned_gt):
                        steps[0] = "A: " + steps[0]
                        token_steps, token_labels = self._get_token_labels_from_step_labels(steps, labels)
                        ## make zero labels are not followed by 1's
                        assert self._check_stitch_labels(token_labels)
                        examples.append((question_tokens, token_steps, token_labels))

        return examples

    def _check_stitch_labels(self, labels):
        '''
        Checks that labels are not followed by 1's
        '''
        for i in range(len(labels) - 1):
            if labels[i] == 0 and labels[i+1] == 1:
                return False
        return True

    def _get_steps_labels(self, aligned_traj, aligned_gt):
        '''
        aligned_traj: list of aligned trajectory steps with '_' for missing steps
        aligned_gt: list of aligned ground truth steps with '_' for missing steps

        NO STITCHING done, just labels over steps. 
        Labels are not necessarily starting with 1's like in stitching
        '''
        
        VAR_RE = re.compile(r"{}(\-?[0-9\.]+)".format('='))

        assert len(aligned_traj) == len(aligned_gt)

        steps = []
        labels = []
        for i, (traj_step, gt_step) in enumerate(zip(aligned_traj, aligned_gt)):
            if traj_step == '_' and gt_step != '_': # missing step, add it with label of 1
                steps.append(gt_step)
                labels.append(1)
            
            elif traj_step != '_' and gt_step == '_': # extra step, add it with label of 0
                steps.append(traj_step)
                labels.append(0)
            
            elif traj_step == '_' and gt_step == '_': # both missing, skip
                continue

            else: # both present, check if they are equal
                traj_vars = VAR_RE.findall(traj_step)
                gt_vars = VAR_RE.findall(gt_step)

                if len(traj_vars) == 0 or len(gt_vars) == 0: # UNKNOWN
                    steps.append(traj_step)
                    labels.append(-1)
                    continue
            
                traj_var = traj_vars[-1]
                gt_var = gt_vars[-1]

                try:
                    traj_var_f = float(traj_var)
                    gt_var_f = float(gt_var)
                    if abs(traj_var_f - gt_var_f) < 1e-3: # correct step
                        steps.append(traj_step)
                        labels.append(1)
                    else:
                        steps.append(traj_step)
                        labels.append(0)
                except:
                    steps.append(traj_step)
                    labels.append(-1)

        return steps, labels

    def _get_steps_labels_by_stitching_steps(self, aligned_traj, aligned_gt):
        # stitch steps together and label them
        # labels are 1's for correct steps, 0's for incorrect steps, -1's for unknown steps
        # labels are starting with 1's, i.e. the first step is always correct

        ## aligned traj: A B _ K E
        ## aligned gt:   A B C D E
        ## steps:        A B C K 
        ## labels:       1 1 1 0

        VAR_RE = re.compile(r"{}(\-?[0-9\.]+)".format('='))

        labels = []
        steps = []

        for i, (traj_step, gt_step) in enumerate(zip(aligned_traj, aligned_gt)):

            gt_steps_so_far = [s for s in aligned_gt[:i] if s != '_']
            assert len(steps) == len(labels)
            
            if traj_step == '_' and gt_step != '_': ## missing step, add gt step with label of 1
                steps.append(gt_step)
                labels.append(1)
                
            elif traj_step != '_' and gt_step == '_': ## extra step
                ## assert labels so far are all 1's (correct prefix)
                assert all([label == 1 for label in labels])
                yield steps + [traj_step], labels + [0]
                
                if steps != gt_steps_so_far:
                    yield gt_steps_so_far + [traj_step], [1] * i + [0]
                
                continue

            elif traj_step == '_' and gt_step == '_':
                continue

            else:
                traj_vars = VAR_RE.findall(traj_step)
                gt_vars = VAR_RE.findall(gt_step)

                if len(traj_vars) == 0 or len(gt_vars) == 0:
                    steps.append(gt_step)
                    labels.append(1)
                    continue
            
                traj_var = traj_vars[-1]
                gt_var = gt_vars[-1]

                try:
                    traj_var_f = float(traj_var)
                    gt_var_f = float(gt_var)
                
                except:
                    assert all([label == 1 for label in labels])
                    yield steps + [traj_step], labels + [0]
                    yield gt_steps_so_far + [gt_step], [1] * (i+1)
                    
                    if steps != gt_steps_so_far:
                        yield steps + [gt_step], labels + [1]
                        yield gt_steps_so_far + [traj_step], [1] * i + [0]

                    break

                if abs(traj_var_f - gt_var_f) < 1e-3:
                    steps.append(traj_step)
                    labels.append(1)
                else:
                    assert all([label == 1 for label in labels])
                    yield steps + [traj_step], labels + [0]
                    yield gt_steps_so_far + [gt_step], [1] * (i+1) # augment with gt step
                    
                    if steps != gt_steps_so_far:
                        yield steps + [gt_step], labels + [1]
                        yield gt_steps_so_far + [traj_step], [1] * i + [0]


                    break

    def _get_token_labels_from_step_labels(self, steps, labels):
        token_steps = []
        token_labels = []

        assert all([s.endswith(self.step_delimiter) for s in steps]), f"steps: {steps}"
        assert len(steps) == len(labels)
        for step, label in zip(steps, labels):
            step_tokens = self.tokenizer.encode(step, add_special_tokens=False)
            token_steps.extend(step_tokens)
            token_labels.extend([-1 if self.labeling == 'step' else label] * (len(step_tokens) - 1) + [label])
        
        assert len(token_steps) == len(token_labels)
        return token_steps, token_labels

    def __len__(self):
        return len(self.examples)
    
    def _process_item_enc_dec(self, qn_tokens, ans_tokens, labels):
        # truncate if too long
        #@if len(ans_tokens) > self.max_len:
        #   print("trajectory too long, {} vs. {}".format(len(ans_tokens), self.max_len))
        
        ## truncate if too long
        qn_tokens = qn_tokens[:self.max_len]
        ans_tokens = ans_tokens[:self.max_len]
        labels = labels[:self.max_len]

        assert len(qn_tokens) <= self.max_len, f"question too long: {self.tokenizer.decode(qn_tokens)}"
        assert  len(ans_tokens) <= self.max_len, f"answer too long: {self.tokenizer.decode(ans_tokens)}"
        assert len(ans_tokens) == len(labels), "lengths don't match!"
        
        # attention masks 
        qn_mask = [1] * len(qn_tokens)
        ans_mask = [1] * len(ans_tokens)
        
        enc_input_tokens = th.tensor(qn_tokens)
        dec_input_tokens = th.tensor(ans_tokens)
        enc_input_mask = th.tensor(qn_mask)
        dec_input_mask = th.tensor(ans_mask)
        labels = th.tensor(labels)
        
        return dict(enc_input_ids=enc_input_tokens, dec_input_ids=dec_input_tokens, 
                                enc_attn_mask=enc_input_mask, 
                                dec_attn_mask=dec_input_mask, labels=labels)

    def _process_item_enc(self, qn_tokens, ans_tokens, labels):
        # truncate if too long
        ## combine qn and ans with a [SEP] token
        tokens = qn_tokens + [self.tokenizer.sep_token_id] + ans_tokens
        labels = [-1] * len(qn_tokens) + [-1] + labels
        token_type_ids = [0] * len(qn_tokens) + [1] + [1] * len(ans_tokens)
        
        if len(tokens) > self.max_len:
            print("input too long {} vs. {}, truncating...".format(len(tokens), self.max_len))
            tokens = tokens[:self.max_len]
            labels = labels[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
        
        # pad to self.max_len 
        tokens_padded = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
        labels_padded = labels + [-1] * (self.max_len - len(labels))
        token_type_ids_padded = token_type_ids + [0] * (self.max_len - len(token_type_ids))

        assert len(tokens) == len(labels), "lengths don't match!"
        assert len(tokens) == len(token_type_ids), "lengths don't match!"

        ## attention masks
        mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
        assert len(tokens_padded) == len(mask), "lengths don't match!"
        
        return dict(input_ids=th.tensor(tokens_padded),
                    attention_mask=th.tensor(mask),
                    token_type_ids=th.tensor(token_type_ids_padded),
                    labels=th.tensor(labels_padded)
        )

    def __getitem__(self, idx):
        qn_tokens, ans_tokens, labels = self.examples[idx]
        if self.model_style == 'enc-dec':
            return self._process_item_enc_dec(qn_tokens, ans_tokens, labels)
        elif self.model_style == 'enc':
            return self._process_item_enc(qn_tokens, ans_tokens, labels)
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented!")
        

    def collate_fn(self, batch):
        if self.model_style == 'enc-dec':
            return self.collate_enc_dec(batch)
        elif self.model_style == 'enc':
            return self.collate_enc(batch)
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented!")
    

    def collate_enc_dec(self, batch):

        enc_input_ids = pad_sequence([x['enc_input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        dec_input_ids = pad_sequence([x['dec_input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        enc_attn_mask = pad_sequence([x['enc_attn_mask'] for x in batch], batch_first=True, padding_value=0)
        dec_attn_mask = pad_sequence([x['dec_attn_mask'] for x in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-1)

        return dict(enc_input_ids=enc_input_ids, dec_input_ids=dec_input_ids,
                    enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask,
                    labels=labels)

    def collate_enc(self, batch):
        input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([x['token_type_ids'] for x in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-1)

        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids, labels=labels)
    
class GSMPairwiseRankingDataset(th.utils.data.Dataset):
    '''
    Dataset that returns a pair of positive and negative trajectory suffixes for each question. 
    Examples: Question, Prefix + Positive Step, Prefix + Negative Step
    Only valid for stitching. 
    '''

    def __init__(self, samples, tokenizer, args=None):
        '''
        Args:
            samples: list of dicts with keys: question and values: trajectories, is_correct, gt_sol
            tokenizer: huggingface tokenizer
            max_len: max length of input sequence
        '''
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.step_delimiter = TASK_TO_STEP_DELIMITER[args.task]
        
        if args.step_delimiter is not None:
            print("Overriding step delimiter with {}".format(args.step_delimiter))
            self.step_delimiter = args.step_delimiter
        
        self.args = args
        self.model_style = args.model_style
        self.invalid_prefix_prob = getattr(args, 'invalid_prefix_prob', 0.0)
        self.step_aligner = StepAligner(model=args.step_aligner_model)

        ## if cls and sep are not already added, add them
        if not self.tokenizer.cls_token:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        if not self.tokenizer.sep_token:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        if self.args.task == 'coin_flip' or self.args.task == "tso": ## load nli model
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').cuda()
            roberta.eval()  # disable dropout for evaluation
            def is_entailed(a, b):
                tokens = roberta.encode(a, b)
                prediction_1 = roberta.predict('mnli', tokens).argmax().item()
                tokens = roberta.encode(b, a)
                prediction_2 = roberta.predict('mnli', tokens).argmax().item()
                return prediction_1 == 2 and prediction_2 == 2
            self.is_entailed = is_entailed
        
        print("Building pairwise dataset...")
        self.examples = self.step_align_and_get_labels(self.samples, args)


    def _extract_steps(self, sol):
        '''
        Extracts steps from a solution
        '''
        if self.step_delimiter == '. ':
            steps = sent_tokenize(sol) #.split(self.step_delimiter)
        elif self.step_delimiter == '|':
            steps = [s.strip() for s in sol.split(self.step_delimiter) if s.strip()]    
        elif self.step_delimiter == ';':
            steps = [s.strip() for s in sol.split(self.step_delimiter) if s.strip()]
            steps = [s for s in steps if 'print' not in s]
        else:
            raise NotImplementedError(f"step delimiter {self.step_delimiter} not implemented!")

        if self.step_delimiter != '. ': # ince sent_tokenize already keeps the periods
            steps = [s + ' ' + self.step_delimiter if not s.endswith(self.step_delimiter) else s for s in steps]
        
        return steps
         
    def step_align_and_get_labels(self, samples, args):
        '''
        Performs stepwise alignment between sampled trajectories and ground truth ones
        Returns a list of tuples (tokenized question, tokenized answer, stepwise labels)
        '''
        examples = []
        n_skipped = 0
        for sample in tqdm(samples):
            #print("*************************************************************")
            #print("Question: ", sample['question'])
            question = sample['question']
            if not question.startswith('Q: '):
                question = 'Q: ' + question
            
            gt_sol = sample['gt_sol'].replace('.\n', '\n').replace('\n', self.step_delimiter).strip()
            assert self.step_delimiter in gt_sol, f"Step delimiter {self.step_delimiter} not found in {gt_sol}"
            gt_sol = gt_sol.split(ANS_IDENTIFIER)[0]
            gt_sol = "A: " + gt_sol

            is_correct = sample['is_correct']
            
            question_tokens = self.tokenizer(question, padding=False, add_special_tokens=False)["input_ids"]

            if gt_sol.startswith('A: '):
                gt_sol = gt_sol[3:] # remove the 'A: ' prefix before splitting to steps
                gt_sol = ' '.join(gt_sol.split())

            gold_steps = self._extract_steps(gt_sol)
            correct_sols = [gold_steps]

            if args.use_correct_samples: 
                print("Using correct samples for alignment!!")
                ## use correctly sampled solutions for alignment as well
                for i, traj in enumerate(sample['trajectories']):
                    if  is_correct[i]:
                        traj = traj.replace('A:', '').strip()
                        traj = traj.replace('.\n', '\n').replace('\n', self.step_delimiter)
                        traj = ' '.join(traj.split())
                        #assert self.step_delimiter in traj, f"Step delimiter {self.step_delimiter} not found in {traj}"
                        traj = traj.split(ANS_IDENTIFIER)[0]
                        if ANS_IDENTIFIER in traj:
                            traj = traj.split(ANS_IDENTIFIER)[0]
                        traj_steps = self._extract_steps(traj)
                        if not traj_steps:
                            continue
                        correct_sols.append(traj_steps)

            ## go over incorrect solutions and align them with correct ones. 
            for i, traj in enumerate(sample['trajectories']):
                traj = traj.replace('A:', '').strip()
                traj = traj.replace('.\n', '\n').replace('\n', self.step_delimiter)
                traj = ' '.join(traj.split())

                if is_correct[i]:
                    continue

                traj = traj.split(ANS_IDENTIFIER)[0]
                if ANS_IDENTIFIER in traj:
                    traj = traj.split(ANS_IDENTIFIER)[0]

                traj_steps = self._extract_steps(traj)
                
                if not traj_steps:
                    continue
                
                for correct_steps in correct_sols:
                    ## make sure every traj step end with step delimiter
                    assert not traj_steps[0].startswith("A: ") and not correct_steps[0].startswith("A: ") # make sure the first step does not start with 'A: ' to not affect the alignment algorithm
                    if getattr(args, 'skip_alignment', False):
                        aligned_traj = traj_steps
                        aligned_correct = correct_steps
                        if len(aligned_traj) != len(aligned_correct):
                            n_skipped += 1
                            continue
                    else:
                        aligned_traj, aligned_correct, cost = self.step_aligner.compute_alignment_from_trajectories(traj_steps, correct_steps, delimiter=self.step_delimiter)
                        if cost > args.max_alignment_cost: ## roughly three gaps
                            n_skipped += 1
                            continue                        
                    
                    for prefix, positive_step, negative_step in self._get_pairwise_examples_by_stitching_trajectories(aligned_traj, aligned_correct):
                        if not prefix.startswith('A: '):
                            prefix = "A: " + prefix
                        prefix_tokens = self.tokenizer(prefix, padding=False, add_special_tokens=False)["input_ids"]
                        positive_step_tokens = self.tokenizer(positive_step, padding=False, add_special_tokens=False)["input_ids"]
                        negative_step_tokens = self.tokenizer(negative_step, padding=False, add_special_tokens=False)["input_ids"]
                        examples.append((question_tokens, prefix_tokens, positive_step_tokens, negative_step_tokens))

        print("Skipping {} trajectories with high alignment cost > {}".format(n_skipped, args.max_alignment_cost))
        return examples
    
    def _get_pairwise_examples_by_stitching_trajectories(self, aligned_traj, aligned_gt):
        labels = []
        steps = []

        for i, (traj_step, gt_step) in enumerate(zip(aligned_traj, aligned_gt)):
            gt_steps_so_far = [s for s in aligned_gt[:i] if s != '_']

            if traj_step == '_' and gt_step != '_': ## missing step, add gt step to the prefix
                steps.append(gt_step)
                labels.append(1)
                
            elif traj_step != '_' and gt_step == '_': ## extra step
                ### get the next ground truth step that is not '_' to use as positive step
                next_gt_step = [s for s in aligned_gt[i:] if s != '_']
                
                if len(next_gt_step) == 0: ## TODO: MAYBE ADD A STOP TOKEN? 
                    break # no more potential positive steps
                
                next_gt_step = next_gt_step[0]
                yield self._create_pairwise_example(prefix=steps, positive_step=next_gt_step, negative_step=traj_step)
                
                if self.args.break_after_extra_step:
                    break
                
            elif traj_step == '_' and gt_step == '_':
                continue

            else:
                ## two aligned steps
                if not self._is_correct_step(traj_step=traj_step, gt_step=gt_step, prefix=steps):
                    if random.random() < self.invalid_prefix_prob:
                        ## allow invalid prefix
                        steps.append(traj_step)                        
                        labels.append(0)
                        continue
                    else:
                        yield self._create_pairwise_example(prefix=steps, positive_step=gt_step, negative_step=traj_step)

                        if steps != gt_steps_so_far:
                            yield self._create_pairwise_example(prefix=gt_steps_so_far, positive_step=gt_step, negative_step=traj_step)
                        break
                else:
                    steps.append(traj_step)
                    labels.append(1)
    
    def _execute_and_get_var(self, program, var_name):
        env = {}
        with timeout(1, program):
            try:
                exec(program, globals(), env)
            except:
                return None
        return env.get(var_name, None)
    
    def _is_correct_step(self, traj_step, gt_step, prefix=None):
        
        if self.args.task in ['gsm8k', 'svamp', 'multiarith']: # math-natural language tasks
            VAR_RE_EQ = re.compile(r"{}(\-?[0-9\.]+)".format('=')) # = xx 
            VAR_RE_NUM = re.compile(r"\d+[.,]?\d*") # any number in the step (to be used when no equation is present, just a number)
            
            traj_vars = VAR_RE_EQ.findall(traj_step)
            gt_vars = VAR_RE_EQ.findall(gt_step)
            
            if len(traj_vars) == 0:
                #try to find a number in the step instead
                traj_vars = VAR_RE_NUM.findall(traj_step)

            if len(gt_vars) == 0:
                gt_vars = VAR_RE_NUM.findall(gt_step)
            
            if len(traj_vars) == 0 or len(gt_vars) == 0: ## no way to compare the two steps -- so we assume traj is incorrect
                return False # need to run an blation with that correct if len(gt_vars) is 0. 
            
            traj_var = traj_vars[-1] # value not variable
            gt_var = gt_vars[-1]
            try:
                traj_var_f = float(traj_var)
                gt_var_f = float(gt_var)
            except:
                return False
            if abs(traj_var_f - gt_var_f) < 1e-3:
                return True
            else:
                return False
        
        elif self.args.task in ['mathqa', 'asdiv']: ## math-to-code tasks
            ## in the form of VAR = VALUE, VAR is alphaneumeric, VALUE is numeric
            if traj_step.strip() == gt_step.strip():
                return True # exact string, so no need to run the program
        
            VAR_RE = re.compile(r"([a-zA-Z0-9]+)")
            traj_vars = VAR_RE.findall(traj_step)
            gt_vars = VAR_RE.findall(gt_step)

            if len(traj_vars) == 0 or len(gt_vars) == 0:
                return False

            traj_var = traj_vars[0]
            gt_var = gt_vars[0]

            prefix_traj = " ".join(prefix + [traj_step])
            prefix_gt = " ".join(prefix + [gt_step])

            traj_var_val = self._execute_and_get_var(prefix_traj, traj_var)
            gt_var_val = self._execute_and_get_var(prefix_gt, gt_var)

            if traj_var_val is None or gt_var_val is None:
                return False
            try:
                return abs(traj_var_val - gt_var_val) < 1e-4
            except:
                return False
        
        elif self.args.task == 'last_letter_concatenation':
            # find strings between double quotes
            VAR_RE = re.compile(r"\"([a-zA-Z]+)\"")
            traj_vars = VAR_RE.findall(traj_step)
            gt_vars = VAR_RE.findall(gt_step)
            
            if len(traj_vars) == 0 or len(gt_vars) == 0:
                return False
            
            traj_var = traj_vars[-1].strip()
            gt_var = gt_vars[-1].strip()

            if traj_var == gt_var:
                return True

            return False
        
        elif self.args.task == 'coin_flip' or self.args.task == 'tso':
            pred = self.is_entailed(traj_step, gt_step)
            return pred
            #if 'head' in traj_step and 'tail' in gt_step:
            #    return False
            #elif 'tail' in traj_step and 'head' in gt_step:
            #    return False
            
            #return True
        
        else: 
            raise NotImplementedError("is_correct_step not implemented for task {}".format(self.args.task))

    def _create_pairwise_example(self, prefix, positive_step, negative_step):
        #positive_traj = " ".join(prefix  + [positive_step])
        #negative_traj = " ".join(prefix + [negative_step])
        prefix = " ".join(prefix)
        return prefix, positive_step, negative_step
    
    def __len__(self):
        return len(self.examples)
    
    def _process_item_enc_dec(self, qn_tokens, ans_tokens, labels):
        raise NotImplementedError("Not implemented yet")

    def _process_item_enc(self, qn_tokens, prefix_tokens, pos_tokens, neg_tokens):
        '''
        Args:
            qn_tokens: list of token ids for the question
            prefix_tokens: list of token ids for the solution prefix
            pos_tokens: list of token ids for the positive step
            neg_tokens: list of token ids for the negative step
        '''
        ## combine [CLS] + Question + Prefix + [SEP] + Positive/Negative Step
        qn_pos_tokens = [self.tokenizer.cls_token_id] + qn_tokens + prefix_tokens + [self.tokenizer.sep_token_id] + pos_tokens
        qn_neg_tokens = [self.tokenizer.cls_token_id] + qn_tokens + prefix_tokens + [self.tokenizer.sep_token_id] + neg_tokens

        pos_token_type_ids = [0] + [0] * len(qn_tokens + prefix_tokens) + [0] + [1] * len(pos_tokens)
        neg_token_type_ids = [0] + [0] * len(qn_tokens + prefix_tokens) + [0] + [1] * len(neg_tokens)

        assert len(qn_pos_tokens) == len(pos_token_type_ids)
        assert len(qn_neg_tokens) == len(neg_token_type_ids)

        qn_pos_attention_mask = [1] * len(qn_pos_tokens)
        qn_neg_attention_mask = [1] * len(qn_neg_tokens)
        
        if len(qn_pos_tokens) > self.max_len:
            print("trajectory too long, {} vs. {}".format(len(qn_pos_tokens), self.max_len))
            qn_pos_tokens = qn_pos_tokens[:self.max_len]
            pos_token_type_ids = pos_token_type_ids[:self.max_len]
            qn_pos_attention_mask = qn_pos_attention_mask[:self.max_len]
        
        if len(qn_neg_tokens) > self.max_len:
            print("trajectory too long, {} vs. {}".format(len(qn_neg_tokens), self.max_len))
            qn_neg_tokens = qn_neg_tokens[:self.max_len]
            neg_token_type_ids = neg_token_type_ids[:self.max_len]
            qn_neg_attention_mask = qn_neg_attention_mask[:self.max_len]
        
        return dict(pos_input_ids=th.tensor(qn_pos_tokens, dtype=th.long),
                    pos_token_type_ids=th.tensor(pos_token_type_ids, dtype=th.long),
                    neg_input_ids=th.tensor(qn_neg_tokens, dtype=th.long),
                    neg_token_type_ids=th.tensor(neg_token_type_ids, dtype=th.long), 
                    pos_attention_mask=th.tensor(qn_pos_attention_mask, dtype=th.long),
                    neg_attention_mask=th.tensor(qn_neg_attention_mask, dtype=th.long))
    
    def __getitem__(self, idx):
        qn_tokens, prefix_ids, positive_ex_ids, negative_ex_ids = self.examples[idx]
        return self._process_item_enc(qn_tokens, prefix_ids, positive_ex_ids, negative_ex_ids)
        

    def collate_fn(self, batch):
        return self.collate_enc(batch)

    def collate_enc(self, batch):

        pos_input_ids = pad_sequence([x['pos_input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        pos_token_type_ids = pad_sequence([x['pos_token_type_ids'] for x in batch], batch_first=True, padding_value=0)
        neg_input_ids = pad_sequence([x['neg_input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        neg_token_type_ids = pad_sequence([x['neg_token_type_ids'] for x in batch], batch_first=True, padding_value=0)

        pos_attention_mask = pad_sequence([x['pos_attention_mask'] for x in batch], batch_first=True, padding_value=0)
        neg_attention_mask = pad_sequence([x['neg_attention_mask'] for x in batch], batch_first=True, padding_value=0)

        return dict(pos_input_ids=pos_input_ids, pos_token_type_ids=pos_token_type_ids,
                    neg_input_ids=neg_input_ids, neg_token_type_ids=neg_token_type_ids, 
                    pos_attention_mask=pos_attention_mask, neg_attention_mask=neg_attention_mask)
    
    
