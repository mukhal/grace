from tqdm import tqdm
import torch
import re, time
from data_utils.utils import use_calculator
from constants import *
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import top_k_top_p_filtering
from torch.nn.utils.rnn import pad_sequence

DELIMITER_TO_ID_T5 = {
    '|': 1820,
    ';': 117,
    '. ': 5
}

DELIMITER_TO_ID_LLAMA = {
    '|': 891,
    '. ': 29889,
    ';': 29936
}

def predict_reasoning(model, 
            model_tokenizer, 
            discriminator, 
            disc_tokenizer,
            model_input_text, 
            disc_input_text, 
            precondition_topk=200, 
            do_sample=False, 
            length_cutoff=150, 
            condition_lambda=1.0,
            use_calculator=False,
            generation_type='token',
            args=None
            ):
    """
    Generates a reasoning path for a given input text with the help of the discriminator.

    Args:
        model: the model to use for generation
        tokenizer: the tokenizer to use for generation (MUST BE same for model and discriminator)
        discriminator: the discriminator to use for reasoning
        model_input_text: the input text to generate a reasoning path for
        disc_input_text: the input text to use for the discriminator (if None, use model_input_text)
        precondition_topk: the number of topk tokens to use for bayes factorization
        do_sample: whether to sample from the topk tokens or take the top one (not implemented)
        length_cutoff: the maximum length of the reasoning path
        condition_lambda: the lambda value to use for bayes factorization
    """

    if not isinstance(model_input_text, list):
        model_input_text = [model_input_text]
    
    if not isinstance(disc_input_text, list):
        disc_input_text = [disc_input_text]

    is_enc_dec = hasattr(model, 'get_encoder')

    with torch.no_grad():
        batch_size = len(model_input_text)
        # assumes initially all same length.
        model_input_ids = [model_tokenizer.encode(it, return_tensors='pt').to(model.device) for it in model_input_text] # batch x seq
        model_input_ids = torch.cat(model_input_ids, dim=0)

        input_ids = torch.LongTensor([[model_tokenizer.pad_token_id]] * len(model_input_text)).to(model.device) # batch x seq
        cur_len = 1
        max_length = length_cutoff
        min_length = 0
        temperature = args.temperature
        top_k = precondition_topk
        top_p = getattr(args, 'top_p', 1.0)
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = [[model_tokenizer.pad_token_id]]
        pad_token_id = model_tokenizer.pad_token_id
        eos_token_id = model_tokenizer.eos_token_id
        attention_mask = model_input_ids.new_ones(model_input_ids.shape)
        use_cache = True
        
        if is_enc_dec:
            model_specific_kwargs = {'encoder_outputs': model.get_encoder()(model_input_ids, attention_mask=attention_mask)}
        else:
            model_specific_kwargs = {}
        
        #### prepare discriminator input (if needed) ####
        if generation_type in ['token', 'step']:
            disc_input_ids = [disc_tokenizer.encode(it, return_tensors='pt').to(discriminator.device) for it in disc_input_text]
            disc_input_ids = torch.cat(disc_input_ids, dim=0)
            disc_attention_mask = disc_input_ids.new_ones(disc_input_ids.shape)

        #### step delimiter depending on model type ####
        step_delimiter = getattr(args, 'step_delimiter', '|')
        if 'T5' in model_tokenizer.__class__.__name__:
            step_delimiter_id = DELIMITER_TO_ID_T5[step_delimiter]
        elif 'Llama' in model_tokenizer.__class__.__name__:
            step_delimiter_id = DELIMITER_TO_ID_LLAMA[step_delimiter]
        else:
            raise NotImplementedError(f"step delimiter ID not set for {model_tokenizer.__class__.__name__}")

        assert step_delimiter not in model_tokenizer.all_special_tokens, "step delimiter cannot be a special token!!!"

        if generation_type == 'token': # TODO refactor this
            output = _generate_no_beam_search_token(model=model,
                        discriminator=discriminator,
                        condition_lambda=condition_lambda,
                        precondition_topk=precondition_topk,
                        decoder_input_ids=input_ids,
                        model_input_ids=model_input_ids,
                        cur_len=cur_len,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        step_delimiter_id=step_delimiter_id,
                        batch_size=batch_size,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        model_kwargs=model_specific_kwargs,
                        disc_input_ids=disc_input_ids,
                        disc_attention_mask=disc_attention_mask,
                        tokenizer=model_tokenizer,
                        use_calculator=use_calculator,
                        args=args)
            
        elif generation_type == 'step':
            output = _generate_step_with_binary_disc(model=model,
                        discriminator=discriminator,
                        condition_lambda=condition_lambda,
                        precondition_topk=precondition_topk,
                        decoder_input_ids=input_ids,
                        model_input_ids=model_input_ids,
                        cur_len=cur_len,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        step_delimiter_id=step_delimiter_id,
                        batch_size=batch_size,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        model_kwargs=model_specific_kwargs,
                        disc_input_ids=disc_input_ids,
                        disc_attention_mask=disc_attention_mask,
                        tokenizer=disc_tokenizer,
                        use_calculator=use_calculator,
                        args=args)
        
        elif generation_type == 'step-score': 
            output = _generate_step_with_score_disc(model=model,
                        discriminator=discriminator,
                        condition_lambda=condition_lambda,
                        precondition_topk=precondition_topk,
                        decoder_input_ids=input_ids,
                        model_input_ids=model_input_ids,
                        cur_len=cur_len,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        step_delimiter_id=step_delimiter_id,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        model_kwargs=model_specific_kwargs,
                        question = disc_input_text[0],
                        model_tokenizer=model_tokenizer,
                        disc_tokenizer=disc_tokenizer,
                        args=args)
        
        elif generation_type == 'step-score-batch': 
            output = _batch_generate_step_with_score_disc(model=model,
                        discriminator=discriminator,
                        condition_lambda=condition_lambda,
                        precondition_topk=precondition_topk,
                        decoder_input_ids=input_ids,
                        model_input_ids=model_input_ids,
                        cur_len=cur_len,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        step_delimiter_id=step_delimiter_id,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        model_kwargs=model_specific_kwargs,
                        question = disc_input_text,
                        model_tokenizer=model_tokenizer,
                        disc_tokenizer=disc_tokenizer,
                        args=args)
        
        elif generation_type == 'step-qrs':
            output = _generate_step_with_score_disc_qrs(model=model,
                        discriminator=discriminator,
                        condition_lambda=condition_lambda,
                        precondition_topk=precondition_topk,
                        decoder_input_ids=input_ids,
                        model_input_ids=model_input_ids,
                        cur_len=cur_len,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        step_delimiter_id=step_delimiter_id,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        model_kwargs=model_specific_kwargs,
                        question = disc_input_text[0],
                        model_tokenizer=model_tokenizer,
                        disc_tokenizer=disc_tokenizer,
                        args=args)
        
        return [model_tokenizer.decode(s, skip_special_tokens=True) for s in output]


# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search_token(
        model,
        discriminator,
        condition_lambda,
        precondition_topk,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
        disc_input_ids,
        disc_attention_mask,
        tokenizer=None,
        use_calculator=False,
        args=None
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
                
        
        ## encode disc_input_ids using the discrimiantor encoder to avoid re-computing 

        disc_input_ids = disc_input_ids.to(discriminator.device)
        disc_attention_mask = disc_attention_mask.to(discriminator.device)
        disc_encoder_outputs = discriminator.t5.encoder(input_ids=disc_input_ids, 
                                                        attention_mask=disc_attention_mask)

        disc_encoder_outputs = disc_encoder_outputs['last_hidden_state'] # 1 x seq x dim
        disc_encoder_outputs  = disc_encoder_outputs.repeat(precondition_topk, 1, 1)
        
        ## repeat the discriminator input TODO: optimize this
        #disc_input_ids = disc_input_ids.repeat(1, precondition_topk).view(-1, disc_input_ids.shape[-1])
        #disc_attention_mask = disc_attention_mask.repeat(1, precondition_topk).view(-1, disc_attention_mask.shape[-1])

        CALC_TOKENS = [tokenizer.encode(s, add_special_tokens=False)[0] for s in ['<<', '>>']]
        assert tokenizer.unk_token_id not in CALC_TOKENS, "Calculator tokens are not identified by the tokenizer!"
        EQUAL_TOKEN = tokenizer.encode('=', add_special_tokens=False)[0]
        
        
        past = None
        while cur_len < max_length:
            
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            ) # TODO Check if attention mask is correct -- seems like it is not

            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            config = GenerationConfig(
                    repetition_penalty=repetition_penalty,
                    encoder_no_repeat_ngram_size=0,
                    forced_bos_token_id=None,
                    forced_eos_token_id=None,
                    num_beam_groups=1,
                    diversity_penalty=0.0,
                    remove_invalid_values=None,
                    exponential_decay_length_penalty=None,
                    renormalize_logits=False,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    max_length=max_length,
                    min_length=min_length,
                    eos_token_id=eos_token_id,
                    num_beams=1,
            )
            
            logits_processor = model._get_logits_processor(
                generation_config = config,
                input_ids_seq_length=input_ids.shape[-1],
                encoder_input_ids=None,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
            )

            scores = logits_processor(input_ids=input_ids, scores=next_token_logits)
            #scores = scores / (temperature if temperature > 0 else 1.0)  # scale by temperature

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            
            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # batch x topk
            ## softmax the logits
            top_logprobs = top_logits #torch.log_softmax(top_logits, dim=1) # batch x topk

            tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1

            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else: 
                tplus1_candidates = tplus1_candidates.view(-1, tplus1_candidates.shape[-1])
                assert disc_encoder_outputs.shape[0] == tplus1_candidates.shape[0]

                ## move to discriminator device
                #disc_input_ids = disc_input_ids.to(discriminator.device)
                #disc_attention_mask = disc_attention_mask.to(discriminator.device)
                disc_encoder_outputs = disc_encoder_outputs.to(discriminator.device)
                tplus1_candidates = tplus1_candidates.to(discriminator.device)

                condition_logits, _ = discriminator.forward_decoder_only(
                                encoder_outputs=disc_encoder_outputs,
                                dec_input_ids=tplus1_candidates, # batch*topk x seq+1
                                shift_right=False
                )
                                
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1)[:, :, -1] # batch x topk of whole prefix 

                ## apply sigmoid to get p(C=1 | prefix)
                condition_logits = torch.log(torch.sigmoid(condition_logits))
                ## renormalize
                #condition_logits = torch.log_softmax(condition_logits, dim=1) # batch x topk
            
            full_logits = (1 - condition_lambda) * top_logprobs + condition_lambda * condition_logits.to(top_logprobs.device) # batch x topk

            ## print topk tokens 
            #K = 10
            #_, top_k_full_indices = full_logits.topk(K, dim=1) # batch x topk
            #top_k_full_indices = top_k_full_indices.cpu().numpy()[0]
            ##
            #_, top_k_condition_indices = condition_logits.topk(K, dim=1) # batch x topk
            #top_k_condition_indices = top_k_condition_indices.cpu().numpy()[0]
            ## 
            #_, top_k_org_indices = top_logits.topk(precondition_topk, dim=1) # batch x topk
            #top_k_org_indices = top_k_org_indices.cpu().numpy()[0]

            ## print topk tokens
            #print("topk tokens")
            #print(f"Top {precondition_topk} original logit tokens", "[" ,[tokenizer.decode(top_indices[0][id]) for id in top_k_org_indices], "]")
            #print(f"Top {K} full logit tokens", "[" ,[tokenizer.decode(top_indices[0][id]) for id in top_k_full_indices], "]")
            #print(f"Top {K} condition logit tokens", "[" ,[tokenizer.decode(top_indices[0][id]) for id in top_k_condition_indices], "]")
            #if top_k_full_indices[0] != top_k_org_indices[0]:
            #    print("discriminator affected the top full logits")

            if do_sample:
                ### top k/top p sampling
                #probs = torch.softmax(full_logits, dim=1)
                #next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # 
                next_token_logscores = top_k_top_p_filtering(full_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(torch.softmax(next_token_logscores, dim=1), num_samples=1).squeeze(1) # batch

            else:
                # Greedy decoding
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)] # batch


            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

def _generate_step_with_binary_disc(
        model,
        discriminator,
        condition_lambda,
        precondition_topk,
        model_input_ids,
        decoder_input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        step_delimiter_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
        disc_input_ids,
        disc_attention_mask,
        tokenizer=None,
        use_calculator=False,
        args=None
    ):
        """generated solutions using step-wise discriminator that was trained for sequence labeling task (with binary cross entropy loss"""
        # length of generated sentences / unfinished sentences
        ## encode disc_input_ids using the discrimiantor encoder to avoid re-computing 

        disc_input_ids = disc_input_ids.to(discriminator.device)
        disc_attention_mask = disc_attention_mask.to(discriminator.device)
        disc_encoder_outputs = discriminator.t5.encoder(input_ids=disc_input_ids, 
                                                        attention_mask=disc_attention_mask)

        disc_encoder_outputs = disc_encoder_outputs['last_hidden_state'] # 1 x seq x dim
        disc_encoder_outputs  = disc_encoder_outputs.repeat(precondition_topk, 1, 1).to(discriminator.device) # topk x seq x dim

        CALC_TOKENS = [tokenizer.encode(s, add_special_tokens=False)[0] for s in ['<<', '>>']]
        assert tokenizer.unk_token_id not in CALC_TOKENS, "Calculator tokens are not identified by the tokenizer!"

        ### 1. sample candidate next steps from the model with beam search
        ### 2. for each candidate, calculate the discriminator score
        ### 3. sample pick the most likely candidate according to the discriminator score
        ### 4. repeat until the end of the solution

        max_steps = args.max_steps 
        past = None

        aggregation_method = args.disc_step_score_aggregation # "mean" or "max" or formula
        sample_method = args.step_sampling_method

        while cur_len < max_steps:
            
            decoder_input_seq_len = decoder_input_ids.shape[1]
            outputs = model.generate(
                decoder_input_ids=decoder_input_ids,
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False if sample_method == "beam" else True,
                temperature=temperature,
                top_p=top_p if sample_method == "top_p" else 1.0,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=step_delimiter_id,
                early_stopping=True if sample_method == "beam" else False,
                num_beams=precondition_topk if sample_method == "beam" else 1,
                num_return_sequences=precondition_topk,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            sequences = outputs.sequences # batch x seq
            new_sequences = sequences[:, decoder_input_seq_len:] # batch x seq
            #new_steps = tokenizer.batch_decode(new_sequences, skip_special_tokens=True)

            if sample_method == "beam":
                ### get only new tokens
                seq_scores = outputs.sequences_scores
                ## softmax the scores
                seq_scores = torch.log_softmax(seq_scores, dim=-1) # batch x seq
            
            elif sample_method in ["top_p", "top_k"]:
                raise NotImplementedError("Top p and top k stepwise sampling not implemented yet!")
                #### zeros 
                #transition_scores = model.compute_transition_scores(
                #    outputs.sequences,
                #    outputs.scores,
                #    normalize_logits=True)
                #seq_scores = torch.zeros(sequences.shape[0]).to(sequences.device) # batch x seq
            
            if condition_lambda > 0.0:
                ## calculate the discriminator score for each sequence
                #assert _sequences.shape[1] == decoder_input_seq_len + new_sequences.shape[1], "Sequence length mismatch!"
                assert disc_encoder_outputs.shape[0] == precondition_topk, "Batch size mismatch!"

                ## feed to discriminator to obtain a sequence of scores
                disc_seq_scores, _ = discriminator.forward_decoder_only(
                    encoder_outputs = disc_encoder_outputs,
                    dec_input_ids = sequences.to(discriminator.device),
                    shift_right=False
                ) # batch x seq

                ### extract score corresponding to the new step sequence only
                disc_seq_scores = disc_seq_scores[:, decoder_input_seq_len:] # batch x seq
                ## apply sigmoid to the scores TODO: check if this is necessary
                disc_seq_scores = torch.sigmoid(disc_seq_scores) # batch x seq
                disc_seq_preds = (disc_seq_scores > 0.5).float() # batch x seq
                #print("disc_seq_preds", disc_seq_preds)
                
                ## mask out the padding tokens and delimiter 

                sequences_pad_mask = (new_sequences != pad_token_id).float().to(disc_seq_scores.device) # batch x seq
                
                ## score each sequence by averaging the scores of each token. We are averaging to avoid encouraging long steps.
                if aggregation_method == "mean":
                    disc_scores = torch.sum(disc_seq_scores * sequences_pad_mask, dim=1) / torch.sum(sequences_pad_mask, dim=1) # batch
                
                elif aggregation_method == "max":
                    disc_scores = torch.max(disc_seq_scores * sequences_pad_mask, dim=1)[0]
                
                elif aggregation_method == "delimiter":
                    delimiter_mask = (new_sequences == step_delimiter_id).to(disc_seq_scores.device) # batch x seq
                    ## if there is no delimiter, then return 0 to use the generator (beam/top_p) score instead
                    disc_scores = [] 
                    for i in range(disc_seq_scores.shape[0]):
                        if torch.sum(delimiter_mask[i]) == 0:
                            disc_scores.append(0.0)
                        else:
                            disc_scores.append(torch.sum(disc_seq_scores[i][delimiter_mask[i]]))
                    
                    disc_scores = torch.tensor(disc_scores).to(disc_seq_scores.device)
                
                elif aggregation_method == "formula":
                    ## For each sequence find idx of << and >> tokens. Then sum the scores of the tokens between them.
                    ## If there are no << or >> tokens, then sum all the scores.
                    ## If there are no tokens, then return 0.
                    disc_scores = []
                    for i in range(disc_seq_scores.shape[0]):
                        seq = new_sequences[i]
                        seq = seq[seq != pad_token_id]
                        if len(seq) == 0:
                            disc_scores.append(0.0)
                            continue
                        if CALC_TOKENS[0] not in seq or CALC_TOKENS[1] not in seq:
                            disc_scores.append(torch.sum(disc_seq_scores[i]))
                            continue
                        start_idx = (seq == CALC_TOKENS[0]).nonzero(as_tuple=True)[0][0]
                        end_idx = (seq == CALC_TOKENS[1]).nonzero(as_tuple=True)[0][0]
                        disc_scores.append(torch.sum(disc_seq_scores[i][start_idx:end_idx+1]))
                    
                    disc_scores = torch.tensor(disc_scores).to(disc_seq_scores.device)
                else:
                    raise NotImplementedError("Aggregation method {} not implemented!".format(aggregation_method))
                    
                ## softmax the scores
                disc_scores = torch.log_softmax(disc_scores, dim=-1).to(seq_scores.device) # batch

                assert disc_scores.shape == seq_scores.shape, "Discriminator scores shape mismatch!"
                ## calculate the final score for each sequence by combining 
                final_scores = (1 - condition_lambda) * seq_scores + condition_lambda * disc_scores # batch
            
            else:
                final_scores = seq_scores

            ## sample the next step with the highest score
            next_step_idx = torch.argmax(final_scores, dim=-1) # batch x 1
            next_step = new_sequences[next_step_idx] # batch x 1
            ## remove padding from the next step
            next_step = next_step[next_step != pad_token_id] # batch x 1
            ## update the decoder_input_ids
            decoder_input_ids = torch.cat([decoder_input_ids, next_step.unsqueeze(0)], dim=-1) # batch x seq
            ## check of eos token is in the next step
            eos_in_sents = next_step == eos_token_id # batch x 1

            #print("next step:", tokenizer.decode(next_step))
            
            if eos_in_sents.sum() > 0:
                break
            
            cur_len += 1

        return decoder_input_ids


def _generate_step_with_score_disc(
        model,
        discriminator,
        condition_lambda,
        precondition_topk,
        model_input_ids,
        decoder_input_ids,
        cur_len,
        temperature,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        step_delimiter_id,
        attention_mask,
        use_cache,
        model_kwargs,
        question,
        model_tokenizer=None,
        disc_tokenizer=None,
        args=None
    ):
        """generated solutions using step-wise discriminator that was trained as a scoring function"""
        # length of generated sentences / unfinished sentences
        ## encode disc_input_ids using the discrimiantor encoder to avoid re-computing 


        assert disc_tokenizer.cls_token is not None, "Score-based discriminator Tokenizer must have a [CLS] token!"
        assert disc_tokenizer.sep_token is not None, "Score-based discriminator tokenizer must have a [SEP] token!"

        prefix_ids = [] # empty prefix to begin with
        
        CALC_TOKENS = [disc_tokenizer.encode(s, add_special_tokens=False)[0] for s in ['<<', '>>']]
        assert disc_tokenizer.unk_token_id not in CALC_TOKENS, "Calculator tokens are not identified by the discriminator tokenizer!"

        ### 1. sample candidate next steps from the model with beam search
        ### 2. for each candidate, calculate the discriminator score
        ### 3. sample pick the most likely candidate according to the discriminator score
        ### 4. repeat until the end of the solution

        max_steps = args.max_steps 
        sample_method = args.step_sampling_method

        #question_ids = torch.tensor(disc_tokenizer.encode(question, add_special_tokens=False))
        question_ids = disc_tokenizer.encode(question, add_special_tokens=False)
        is_enc_dec = hasattr(model, "get_encoder")

        ## cache encoder_outputs
        if is_enc_dec:
            _encoder_outputs = model.get_encoder()(model_input_ids.repeat_interleave(1, dim=0), return_dict=True)
            _last_hidden_state = _encoder_outputs["last_hidden_state"].clone()
            model_kwargs = {"encoder_outputs": _encoder_outputs}
            cur_prefix = decoder_input_ids
        else: # decoder-only models
            model_kwargs = {}
            cur_prefix = model_input_ids
        
        # #### id
        if 'T5' in model_tokenizer.__class__.__name__:
            AND_IDENTIFIER_ID = 30345 
        elif 'Llama' in model_tokenizer.__class__.__name__:
            AND_IDENTIFIER_ID = 3191 
        else:
            raise NotImplementedError("tokenizer {} not supported!".format(model_tokenizer.__class__.__name__))
        
        original_input_length = cur_prefix.shape[1]
        
        for _ in tqdm(range(max_steps), disable=True): 
            
            decoder_input_seq_len = cur_prefix.shape[1]
            
            all_new_sequences = [] 
            all_seq_scores = []
            sampling_bsz = getattr(args, "sampling_batch_size", precondition_topk)

            for i in range(0, precondition_topk, sampling_bsz):
                n_to_sample = min(sampling_bsz, precondition_topk - i)

                outputs = model.generate(
                    decoder_input_ids=cur_prefix if is_enc_dec else None,
                    input_ids=model_input_ids if is_enc_dec else cur_prefix,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_step_length,
                    do_sample=False if sample_method == "beam" else True,
                    temperature=temperature,
                    top_p=args.top_p if sample_method == "top_p" else 1.0,
                    top_k=args.top_k if sample_method == "top_k" else None,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=[step_delimiter_id],
                    num_beams=n_to_sample if sample_method == "beam" else 1,
                    num_return_sequences=n_to_sample,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=use_cache,
                    sample_calc=args.sample_calc,
                    tokenizer=model_tokenizer,
                    **model_kwargs,
                )
                
                if is_enc_dec:
                    model_kwargs["encoder_outputs"]["last_hidden_state"] = _last_hidden_state.clone() ## reset the encoder outputs to have a batch size of 1
                
                sequences = outputs.sequences # batch x seq
                new_sequences = sequences[:, decoder_input_seq_len:] # batch x seq

                if sample_method == "beam":
                    ### get only new tokens
                    seq_scores = outputs.sequences_scores
                    ## softmax the scores
                    seq_scores = torch.softmax(seq_scores, dim=-1) # batch x seq
                
                elif sample_method in ["top_p", "top_k", "random"]:
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences,
                        outputs.scores,
                        normalize_logits=True)# batch x seq
                    
                    assert transition_scores.shape[1] == new_sequences.shape[1], "Transition scores and sequences mismatch!"

                    ## normalize by length: exp()
                    probs = torch.exp(transition_scores) # batch x seq
                    logprobs = torch.log(probs) # batch x seq
                    ## divide by length of each sequence
                    seq_lens = torch.sum(new_sequences != pad_token_id, dim=-1).unsqueeze(-1) # batch x 1
                    logprobs = logprobs / seq_lens # batch x seq
                    ### set -inf to 0
                    logprobs[logprobs == float('-inf')] = 0.0
                    seq_scores = torch.exp(torch.sum(logprobs, dim=-1))
                    #seq_scores = torch.ones(sequences.shape[0]).to(sequences.device) # batch x seq

                all_new_sequences.extend(new_sequences)
                all_seq_scores.append(seq_scores)

            new_sequences = pad_sequence(all_new_sequences, batch_first=True, padding_value=pad_token_id)
            seq_scores = torch.cat(all_seq_scores, dim=0)

            assert new_sequences.shape[0] == seq_scores.shape[0], "new_sequences and seq_scores mismatch!"
            assert new_sequences.shape[0] == precondition_topk, "new_sequences and precondition_topk mismatch!"

            ### check if all sequences contain a final answer
            is_all_answers = torch.all(torch.sum(new_sequences == AND_IDENTIFIER_ID, dim=1))

            if condition_lambda > 0.0 and not is_all_answers:
                disc_input_ids = []
                prefix_ids_disc = prefix_ids

                if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                    prefix_ids_disc = model_tokenizer.decode(prefix_ids_disc, skip_special_tokens=True)
                    prefix_ids_disc = disc_tokenizer.encode(prefix_ids_disc, add_special_tokens=False)
                
                for seq in new_sequences:
                    seq = seq.tolist()
                    ### if the two tokenizers are different, we need to convert the sequence to the discriminator tokenizer
                    if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                        seq = model_tokenizer.decode(seq, skip_special_tokens=True)
                        seq = disc_tokenizer.encode(seq, add_special_tokens=False)
                    
                    disc_input_ids.append([disc_tokenizer.cls_token_id] + question_ids + prefix_ids_disc + [disc_tokenizer.sep_token_id] + seq)
                
                ## pad the sequences
                disc_input_ids = pad_sequence([torch.tensor(t) for t in disc_input_ids], batch_first=True, padding_value=disc_tokenizer.pad_token_id).to(discriminator.device) # batch x seq
                disc_attention_mask = disc_input_ids != disc_tokenizer.pad_token_id # batch x seq
                ## feed to discriminator to obtain scores
                disc_scores = discriminator.forward_scores(input_ids=disc_input_ids, attention_mask=disc_attention_mask).view(-1)
                
                if args.normalize_disc_scores:
                    disc_scores = torch.softmax(disc_scores, dim=-1).to(seq_scores.device) # batch
                
                assert disc_scores.shape == seq_scores.shape, "Discriminator scores shape mismatch!"
                ## calculate the final score for each sequence by combining 
                final_scores = (1 - condition_lambda) * seq_scores + condition_lambda * disc_scores # batch
            
            else:
                final_scores = seq_scores

            if args.step_selection_method == "greedy":
                ## sample the next step with the highest score
                next_step_idx = torch.argmax(final_scores, dim=-1) # batch x 1

            elif args.step_selection_method == "sample":
                ## sample the next step with probability proportional to the score
                next_step_idx = torch.multinomial(final_scores, num_samples=1)
            else:
                raise ValueError("Invalid step selection method!")
            
            next_step = new_sequences[next_step_idx] # batch x 1
            ## remove padding from the next step
            next_step = next_step[next_step != pad_token_id] # batch x 1
            ## update the decoder_input_ids
            cur_prefix = torch.cat([cur_prefix, next_step.unsqueeze(0)], dim=-1) # batch x seq
            ## update attention mask if necessary
            if not is_enc_dec:
                attention_mask = cur_prefix != pad_token_id # batch x seq
            
            ## check of eos token is in the next step
            eos_in_sents = next_step == eos_token_id # batch x 1
            prefix_ids += next_step.tolist()

            #print("Generated step: ", model_tokenizer.decode(next_step.tolist(), skip_special_tokens=True))
            
            if eos_in_sents.sum() > 0 or is_all_answers or AND_IDENTIFIER_ID in next_step.tolist():
                break

            cur_len += 1

        if not is_enc_dec: # remove the input prefix from the generated sequence
            cur_prefix = cur_prefix[:, original_input_length:]
        
        return cur_prefix


def _generate_step_with_score_disc_qrs(
        model,
        discriminator,
        condition_lambda,
        precondition_topk,
        model_input_ids,
        decoder_input_ids,
        cur_len,
        temperature,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        step_delimiter_id,
        attention_mask,
        use_cache,
        model_kwargs,
        question,
        model_tokenizer=None,
        disc_tokenizer=None,
        args=None
    ):
        """generated solutions using step-wise discriminator that was trained as a scoring function
        This function uses quasi rejection sampling 
        """
        assert disc_tokenizer.cls_token is not None, "Score-based discriminator Tokenizer must have a [CLS] token!"
        assert disc_tokenizer.sep_token is not None, "Score-based discriminator tokenizer must have a [SEP] token!"

        prefix_ids = [] # empty prefix to begin with
        
        CALC_TOKENS = [disc_tokenizer.encode(s, add_special_tokens=False)[0] for s in ['<<', '>>']]
        assert disc_tokenizer.unk_token_id not in CALC_TOKENS, "Calculator tokens are not identified by the discriminator tokenizer!"

        ### 1. sample candidate next steps from the model with beam search
        ### 2. for each candidate, calculate the discriminator score
        ### 3. sample pick the most likely candidate according to the discriminator score
        ### 4. repeat until the end of the solution

        max_steps = args.max_steps 
        sample_method = args.step_sampling_method

        #question_ids = torch.tensor(disc_tokenizer.encode(question, add_special_tokens=False))
        question_ids = disc_tokenizer.encode(question, add_special_tokens=False)
        is_enc_dec = hasattr(model, "get_encoder")

        ## cache encoder_outputs
        if is_enc_dec:
            _encoder_outputs = model.get_encoder()(model_input_ids.repeat_interleave(1, dim=0), return_dict=True)
            _last_hidden_state = _encoder_outputs["last_hidden_state"].clone()
            model_kwargs = {"encoder_outputs": _encoder_outputs}
            cur_prefix = decoder_input_ids
        else: # decoder-only models
            model_kwargs = {}
            cur_prefix = model_input_ids
        
        # #### id
        if 'T5' in model_tokenizer.__class__.__name__:
            AND_IDENTIFIER_ID = 30345 
        elif 'Llama' in model_tokenizer.__class__.__name__:
            AND_IDENTIFIER_ID = 3191 
        else:
            raise NotImplementedError("tokenizer {} not supported!".format(model_tokenizer.__class__.__name__))

        assert sample_method != "beam", "Beam search is not supported for QRS sampling"
        question_ids = disc_tokenizer.encode(question, add_special_tokens=False)

        original_input_length = cur_prefix.shape[1]
        beta = args.qrs_beta
        sampling_bsz = getattr(args, "sampling_batch_size", precondition_topk)
        max_rejection_tries = getattr(args, "max_rejection_tries", 30)
        
        for _ in tqdm(range(max_steps), disable=True):   
            decoder_input_seq_len = cur_prefix.shape[1]
            n_tries = 0

            while True: ## keep sampling until we accept a step
                outputs = model.generate(
                    decoder_input_ids=cur_prefix if is_enc_dec else None,
                    input_ids=model_input_ids if is_enc_dec else cur_prefix,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_step_length,
                    do_sample=False if sample_method == "beam" else True,
                    temperature=temperature,
                    top_p=args.top_p if sample_method == "top_p" else 1.0,
                    top_k=args.top_k if sample_method == "top_k" else None,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=[step_delimiter_id],
                    num_beams=sampling_bsz if sample_method == "beam" else 1,
                    num_return_sequences = sampling_bsz,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=use_cache,
                    sample_calc=args.sample_calc,
                    tokenizer=model_tokenizer,
                    **model_kwargs,
                )

                if is_enc_dec:
                    model_kwargs["encoder_outputs"]["last_hidden_state"] = _last_hidden_state.clone() ## reset the encoder outputs to have a batch size of 1

                sequences = outputs.sequences # batch x seq
                new_sequences = sequences[:, decoder_input_seq_len:] # batch x seq
                
                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True)# batch x seq
                
                assert transition_scores.shape[1] == new_sequences.shape[1], "Transition scores and sequences mismatch!"

                ## normalize by length: exp()
                probs = torch.exp(transition_scores) # batch x seq
                logprobs = torch.log(probs) # batch x seq
                ## divide by length of each sequence
                seq_lens = torch.sum(new_sequences != pad_token_id, dim=-1).unsqueeze(-1) # batch x 1
                logprobs = logprobs / seq_lens # batch x seq
                ### set -inf to 0
                logprobs[logprobs == float('-inf')] = 0.0
                seq_scores = torch.exp(torch.sum(logprobs, dim=-1))

                ### check if all sequences contain a final answer
                is_all_answers = torch.all(torch.sum(new_sequences == AND_IDENTIFIER_ID, dim=1))

                if condition_lambda > 0.0 and not is_all_answers:
                    disc_input_ids = []
                    prefix_ids_disc = prefix_ids

                    if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                        prefix_ids_disc = model_tokenizer.decode(prefix_ids_disc, skip_special_tokens=True)
                        prefix_ids_disc = disc_tokenizer.encode(prefix_ids_disc, add_special_tokens=False)
                    
                    for seq in new_sequences:
                        seq = seq.tolist()
                        ### if the two tokenizers are different, we need to convert the sequence to the discriminator tokenizer
                        if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                            seq = model_tokenizer.decode(seq, skip_special_tokens=True)
                            seq = disc_tokenizer.encode(seq, add_special_tokens=False)
                    
                        disc_input_ids.append([disc_tokenizer.cls_token_id] + question_ids + prefix_ids_disc + [disc_tokenizer.sep_token_id] + seq)
                
                    ## pad the sequences
                    disc_input_ids = pad_sequence([torch.tensor(t) for t in disc_input_ids], batch_first=True, padding_value=disc_tokenizer.pad_token_id).to(discriminator.device) # batch x seq
                    disc_attention_mask = disc_input_ids != disc_tokenizer.pad_token_id # batch x seq
                    ## feed to discriminator to obtain scores
                    disc_scores = discriminator.forward_scores(input_ids=disc_input_ids, attention_mask=disc_attention_mask).view(-1)

                    disc_scores += 1.00001 # to make it all > 0
                    ## make sure all disc_score > 0
                    assert torch.all(disc_scores >= 0), "Discriminator scores must be > 0!"
                
                    assert disc_scores.shape == seq_scores.shape, "Discriminator scores shape mismatch!"
                    ## calculate the final score for each sequence by combining 
                    acceptance_ratio = disc_scores / (beta * seq_scores)
                    ## compute acceptance probability min(1, disc_score / (beta * seq_score))
                    acceptance_probs = torch.min(torch.ones_like(disc_scores), acceptance_ratio)
                    ## sample from uniform distribution
                    u = torch.rand_like(acceptance_probs)
                    ## accept if u <= acceptance_probs
                    accepted = u <= acceptance_probs
                    
                    if accepted.sum() > 0:
                        ## pick the step with the highest acceptance probability
                        next_step_idx = torch.argmax(acceptance_probs)
                        break
                    else:
                        print("All steps rejected. Resampling...")
                        n_tries += 1
                        if n_tries >= max_rejection_tries:
                            #select the most probable sequence
                            print("Max rejection tries reached. Selecting the most probable sequence anyway...")
                            next_step_idx = torch.argmax(acceptance_probs)
                            break 
                        else:
                            continue
                else:
                    ### pick most probable acording to seq_scores
                    next_step_idx = torch.argmax(seq_scores)
                    break

            next_step = new_sequences[next_step_idx] # batch x 1
            ## remove padding from the next step
            next_step = next_step[next_step != pad_token_id] # batch x 1
            ## update the decoder_input_ids
            cur_prefix = torch.cat([cur_prefix, next_step.unsqueeze(0)], dim=-1) # batch x seq
            
            ## update attention mask if necessary
            if not is_enc_dec:
                attention_mask = cur_prefix != pad_token_id # batch x seq
            
            ## check of eos token is in the next step
            eos_in_sents = next_step == eos_token_id # batch x 1
            prefix_ids += next_step.tolist()

            #print("Generated step: ", model_tokenizer.decode(next_step.tolist(), skip_special_tokens=True))
            
            if eos_in_sents.sum() > 0 or is_all_answers or AND_IDENTIFIER_ID in next_step.tolist():
                break

            cur_len += 1

        if not is_enc_dec: # remove the input prefix from the generated sequence
            cur_prefix = cur_prefix[:, original_input_length:]
        
        return cur_prefix


def _batch_generate_step_with_score_disc(
        model,
        discriminator,
        condition_lambda,
        precondition_topk,
        model_input_ids,
        decoder_input_ids,
        cur_len,
        temperature,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        step_delimiter_id,
        attention_mask,
        use_cache,
        model_kwargs,
        question,
        model_tokenizer=None,
        disc_tokenizer=None,
        args=None
    ):
        """generated solutions using step-wise discriminator that was trained as a scoring function"""
        # length of generated sentences / unfinished sentences
        ## encode disc_input_ids using the discrimiantor encoder to avoid re-computing 


        assert disc_tokenizer.cls_token is not None, "Score-based discriminator Tokenizer must have a [CLS] token!"
        assert disc_tokenizer.sep_token is not None, "Score-based discriminator tokenizer must have a [SEP] token!"

        prefix_ids = [[] for _ in range(model_input_ids.shape[0])]
        
        CALC_TOKENS = [disc_tokenizer.encode(s, add_special_tokens=False)[0] for s in ['<<', '>>']]
        assert disc_tokenizer.unk_token_id not in CALC_TOKENS, "Calculator tokens are not identified by the discriminator tokenizer!"

        ### 1. sample candidate next steps from the model with beam search
        ### 2. for each candidate, calculate the discriminator score
        ### 3. sample pick the most likely candidate according to the discriminator score
        ### 4. repeat until the end of the solution

        max_steps = args.max_steps 
        sample_method = args.step_sampling_method

        #question_ids = torch.tensor(disc_tokenizer.encode(question, add_special_tokens=False))
        question_ids = [disc_tokenizer.encode(q, add_special_tokens=False) for q in question]
        unfinished_solutions = torch.ones(len(question_ids), dtype=torch.long).to(model_input_ids.device)

        is_enc_dec = hasattr(model, "get_encoder")

        ## cache encoder_outputs
        if is_enc_dec:
            _encoder_outputs = model.get_encoder()(model_input_ids.repeat_interleave(1, dim=0), return_dict=True)
            _last_hidden_state = _encoder_outputs["last_hidden_state"].clone()
            model_kwargs = {"encoder_outputs": _encoder_outputs}
            new_inputs = decoder_input_ids
            attention_mask = None
        else: # decoder-only models
            model_kwargs = {}
            new_inputs = model_input_ids
            attention_mask = model_input_ids != pad_token_id
        
        # #### id
        if 'T5' in model_tokenizer.__class__.__name__:
            ANS_IDENTIFIER_ID = 30345 
        elif 'Llama' in model_tokenizer.__class__.__name__:
            ANS_IDENTIFIER_ID = 3191 
        else:
            raise NotImplementedError("tokenizer {} not supported!".format(model_tokenizer.__class__.__name__))
        
        ans_token_id_tensor = torch.tensor([ANS_IDENTIFIER_ID]).to(model_input_ids.device)

        cur_prefix = new_inputs.clone()
        original_input_length = cur_prefix.shape[1]
        
        for _ in tqdm(range(max_steps), disable=True):   
            start_time = time.time()
            decoder_input_seq_len = cur_prefix.shape[1]
            
            outputs = model.generate(
                decoder_input_ids=cur_prefix if is_enc_dec else None,
                input_ids=model_input_ids if is_enc_dec else cur_prefix,
                attention_mask=attention_mask,
                max_new_tokens=args.max_step_length,
                do_sample=False if sample_method == "beam" else True,
                temperature=temperature,
                top_p=args.top_p if sample_method == "top_p" else 1.0,
                top_k=args.top_k if sample_method == "top_k" else None,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=[step_delimiter_id],
                num_beams=precondition_topk if sample_method == "beam" else 1,
                num_return_sequences=precondition_topk,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                sample_calc=args.sample_calc,
                tokenizer=model_tokenizer,
                **model_kwargs,
            )
            
            if is_enc_dec:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = _last_hidden_state.clone() ## reset the encoder outputs to have a batch size of 1

            sequences = outputs.sequences # batch x seq
            new_sequences = sequences[:, decoder_input_seq_len:] # batch x seq
            
            assert sequences.shape[0] == len(question_ids) * precondition_topk, "Batch size mismatch!"

            if sample_method == "beam":
                ### get only new tokens
                seq_scores = outputs.sequences_scores
                ## softmax the scores
                seq_scores = torch.softmax(seq_scores, dim=-1) # batch x seq
            
            elif sample_method in ["top_p", "top_k", "random"]:
                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True)# batch x seq
                
                assert transition_scores.shape[1] == new_sequences.shape[1], "Transition scores and sequences mismatch!"

                ## normalize by length: exp()
                probs = torch.exp(transition_scores) # batch x seq
                logprobs = torch.log(probs) # batch x seq
                ## divide by length of each sequence
                seq_lens = torch.sum(new_sequences != pad_token_id, dim=-1).unsqueeze(-1) # batch x 1
                logprobs = logprobs / seq_lens # batch x seq
                ### set -inf to 0
                logprobs[logprobs == float('-inf')] = 0.0
                seq_scores = torch.exp(torch.sum(logprobs, dim=-1))
                seq_lens = torch.sum(new_sequences != pad_token_id, dim=-1).unsqueeze(-1) # batch * K x 1
                #seq_scores = torch.ones(sequences.shape[0]).to(sequences.device) # batch x seq

            ### check if all sequences contain a final answer
            is_all_answers = torch.all(torch.sum(new_sequences == ANS_IDENTIFIER_ID, dim=1))

            start_time = time.time()
            
            if condition_lambda > 0.0 and not is_all_answers:
                disc_input_ids = []
                prefix_ids_disc = prefix_ids

                if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                    prefix_ids_disc = model_tokenizer.batch_decode(prefix_ids_disc, skip_special_tokens=True)
                    prefix_ids_disc = disc_tokenizer(prefix_ids_disc, add_special_tokens=False, padding=True, truncation=False)["input_ids"]
                
                for i, step in enumerate(new_sequences):
                    seq = step.tolist()
                    q_idx = i // precondition_topk
                    qids = question_ids[q_idx]
                    pids = prefix_ids_disc[q_idx]

                    ### if the two tokenizers are different, we need to convert the sequence to the discriminator tokenizer
                    if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                        seq = model_tokenizer.decode(seq, skip_special_tokens=True)
                        seq = disc_tokenizer.encode(seq, add_special_tokens=False)
                    
                    disc_input_ids.append([disc_tokenizer.cls_token_id] + qids + pids + [disc_tokenizer.sep_token_id] + seq)
                
                ## pad the sequences
                disc_input_ids = pad_sequence([torch.tensor(t) for t in disc_input_ids], batch_first=True, padding_value=disc_tokenizer.pad_token_id).to(discriminator.device) # batch x seq
                disc_attention_mask = disc_input_ids != disc_tokenizer.pad_token_id # batch x seq
                ## feed to discriminator to obtain scores
                disc_scores = discriminator.forward_scores(input_ids=disc_input_ids, attention_mask=disc_attention_mask).view(-1)
                disc_scores = torch.softmax(disc_scores, dim=-1).to(seq_scores.device) # batch
                assert disc_scores.shape == seq_scores.shape, "Discriminator scores shape mismatch!"
                ## calculate the final score for each sequence by combining 
                final_scores = (1 - condition_lambda) * seq_scores + condition_lambda * disc_scores # batch
            else:
                final_scores = seq_scores

            #print("time for discriminator: ", time.time() - start_time)

            ## pick next step for each question 
            q_step_scores = final_scores.view(len(question), precondition_topk) # batch x K
            if args.step_selection_method == "greedy":
                ## sample the next step with the highest score
                next_step_idx = torch.argmax(q_step_scores, dim=-1) # batch x 1

            elif args.step_selection_method == "sample":
                ## sample the next step with probability proportional to the score
                next_step_idx = torch.multinomial(q_step_scores, num_samples=1)
            else:
                raise ValueError("Invalid step selection method!")
            
            q_steps = new_sequences.view(len(question), precondition_topk, new_sequences.shape[1]) # batch x K x seq
            next_steps = torch.gather(q_steps, dim=1, index=next_step_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, q_steps.shape[-1])).squeeze(1) # batch x seq

            past_key_values = outputs.past_key_values # (((batch x num_heads x seq x dim), (batch x num_heads x seq x dim)), ...)
            new_past_key_values = []
            
            for layer_past_key_values in past_key_values:
                key, value = layer_past_key_values
                key = key.view(len(question), precondition_topk, key.shape[1], key.shape[2], key.shape[3]) # batch x K x num_heads x seq x dim
                bsz, k, num_heads, seq_len, dim = key.shape
                value = value.view(len(question), precondition_topk, value.shape[1], value.shape[2], value.shape[3]) # batch x K x num_heads x seq x dim
                new_key = torch.gather(key, dim=1, index=next_step_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_heads, seq_len, dim)).squeeze(1)
                new_value = torch.gather(value, dim=1, index=next_step_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_heads, seq_len, dim)).squeeze(1)

                ## repeat key k times to prepare for next step
                new_key = new_key.repeat_interleave(precondition_topk, dim=0) # batch x K x num_heads x seq x dim
                new_value = new_value.repeat_interleave(precondition_topk, dim=0) # batch x K x num_heads x seq x dim
                new_past_key_values.append((new_key, new_value))
            
            ## repad next steps max new length 
            #max_new_len = torch.max(torch.sum(next_steps != pad_token_id, dim=-1))
            #next_steps = next_steps[:, :max_new_len]
            all_pad_step = torch.tensor([pad_token_id] * next_steps.shape[1]).to(new_sequences.device)

            # if a question is finished use all_pad_step instead. 
            all_next_steps = unfinished_solutions.unsqueeze(1) * next_steps + (1 - unfinished_solutions.unsqueeze(1)) * all_pad_step

            ### conve

            cur_prefix = torch.cat([cur_prefix, all_next_steps], dim=-1) # batch x seq
            
            ## convert cur_prefix to from right padding to left padding
            new_prefixes = []
            for i in range(len(question)):
                pref = cur_prefix[i][cur_prefix[i] != pad_token_id]
                new_prefixes.append(pref)

            ## left padding the prefix instead of right padding
            cur_prefix = pad_sequence(new_prefixes[::-1], batch_first=True, padding_value=pad_token_id).to(new_sequences.device).flip(dims=[1]) # batch x seq
            #attention_mask = torch.cat([attention_mask, all_next_steps != pad_token_id], dim=-1) # batch x seq
            attention_mask = cur_prefix != pad_token_id #TODO I think the current problem is with attention MASK! 
            import ipdb; ipdb.set_trace()

            assert cur_prefix.shape == attention_mask.shape

            print("Next step: ", model_tokenizer.batch_decode(all_next_steps.tolist(), skip_special_tokens=True))
            unfinished_solutions = unfinished_solutions.mul(all_next_steps.ne(ans_token_id_tensor.unsqueeze(1)).prod(dim=1))
            
            if unfinished_solutions.max() == 0:
                break

            for i in range(len(question)):
                ns_nopad = all_next_steps[i][all_next_steps[i] != pad_token_id]
                prefix_ids[i] += ns_nopad.tolist()

            cur_len += 1
            model_kwargs["past_key_values"] = new_past_key_values
            new_inputs = all_next_steps

        
        if not is_enc_dec: # remove the input prefix from the generated sequence
            cur_prefix = cur_prefix[:, original_input_length:]
        
        return cur_prefix




            
        





            
        
