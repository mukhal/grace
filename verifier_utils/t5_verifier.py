import json, os 
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
import math
import torch.nn as nn

class T5Verifier(torch.nn.Module):
    '''
    Disciminator takes a sequence of steps and produces and energy score for the whole sequence. Higher Energy means more likely to be correct.
    IS comprised of a T5 encoder and a linear layer.
    '''
    def __init__(self, model_name_or_path, args=None, device='cuda'):
        super().__init__()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        #is_decoder is set to True to use only consider previous steps in the reasoning chain
        hidden_size = self.model.config.hidden_size
        self.out_linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.out_linear2 = torch.nn.Linear(hidden_size, 2) # correct vs. incorrect
        self.loss_fct = torch.nn.CrossEntropyLoss()        
        self.device = device
        self.to(device)
        self.pooling = args.pooling if hasattr(args, 'pooling') else 'max'

    def forward(self, input_ids, attention_mask, labels=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        hidden_out = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        b, _, h = hidden_out.shape
        ## take max pooling over the sequence dimension over non-padded tokens
        
        if self.pooling == 'max':
            hidden_out = torch.where(attention_mask.unsqueeze(-1) == 0, torch.tensor(-1e9).to(self.device), hidden_out)
            hidden_out = torch.max(hidden_out, dim=1)[0] # (b, h)
        elif self.pooling == 'mean':
            hidden_out = torch.sum(hidden_out * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        elif self.pooling == 'cls':
            hidden_out = hidden_out[:, 0, :]
        
        hidden_out = hidden_out.view(b, h)
        linear_out = torch.relu(self.out_linear1(hidden_out))
        logits = self.out_linear2(linear_out)
                
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss, logits

        return logits

    
