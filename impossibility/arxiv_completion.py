# Create an arXiv completion dataset from the abstracts dataset.

import json
import re
import numpy as np
import os

from tools import progress_bar

from transformers import RobertaTokenizer

arxiv_dataset = 'data/arxiv-abstracts.jsonl'
output_dir = 'data/arxiv-completions-100'
num_abstracts = 10**6
num_train = 20000
num_valid = 5000
num_test = 5000
max_completions = num_train + num_valid + num_test
prompt_size = 100

# Load abstracts
print('Loading first {} abstracts from {}'.format(num_abstracts, arxiv_dataset))
abstracts = []
with open(arxiv_dataset, 'r') as f:
    i = 0
    for line in f:
        i += 1
        if i > num_abstracts:
            break
        data = json.loads(line)
        abstracts.append(data['abstract'])

        # Print progress
        print(' ' * 4 + progress_bar(i / num_abstracts), end='\r')
    
    print('')

# Shuffle data
np.random.shuffle(abstracts)
print('Loaded %d abstracts.' % len(abstracts))

# Create tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create completions
print('Creating completions dataset...')
i = 0
json_list = []
for abs in abstracts:
    abs = re.sub('\s*\n\s*', ' ', abs)
    abs = abs.strip()
    tokens = tokenizer.tokenize(abs)
    if len(tokens) >= prompt_size:
        i += 1
        if i > max_completions:
            break
        prompts = tokens[:prompt_size]
        completions = tokens[prompt_size:]
        prompts = tokenizer.convert_tokens_to_string(prompts).strip()
        completions = tokenizer.convert_tokens_to_string(completions).strip()
        json_list.append({'prompt': prompts, 'completion': completions})

        # Print progress
        print(' ' * 4 + progress_bar(i / max_completions), end='\r')

print('')

print('Created %d samples' % len(json_list))

# Create splits
json_train = json_list[:num_train]
json_valid = json_list[num_train:num_train+num_valid]
json_test = json_list[num_train+num_valid:]

# Save splits
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'arxiv-completion.train.jsonl'), 'w') as f:
    for json_obj in json_train:
        f.write(json.dumps(json_obj) + '\n')

print('Saved {} samples to {}/arxiv-completion.train.jsonl'.format(len(json_train), output_dir))

with open(os.path.join(output_dir, 'arxiv-completion.valid.jsonl'), 'w') as f: # with open('data/arxiv-completion.valid.jsonl', 'w') as f:
    for json_obj in json_valid:
        f.write(json.dumps(json_obj) + '\n')

print('Saved {} samples to {}/arxiv-completion.valid.jsonl'.format(len(json_valid), output_dir))
        

with open(os.path.join(output_dir, 'arxiv-completion.test.jsonl'), 'w') as f: # with open('data/arxiv-completion.test.jsonl', 'w') as f:
    for json_obj in json_test:
        f.write(json.dumps(json_obj) + '\n')

print('Saved {} samples to {}/arxiv-completion.test.jsonl'.format(len(json_test), output_dir))
