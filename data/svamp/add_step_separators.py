import json 
from argparse import ArgumentParser 
from nltk.tokenize import sent_tokenize


def sents_to_steps(s):
    steps = sent_tokenize(s)
    return '\n'.join(steps)

parser = ArgumentParser()
parser.add_argument('--file', type=str)

args = parser.parse_args()

with  open(args.file, 'r') as f:# jsonl
    data = f.readlines()
    data = [json.loads(d) for d in data]

for d in data:
    d['answer'] = sents_to_steps(d['answer'])
    if 'cots' in d:
        cots_steps = [sents_to_steps(c) for c in d['cots']]
        d['cots'] = cots_steps

with open(args.file + '.steps', 'w') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

    
