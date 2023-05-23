# Create a dataset that has the first 100 tokens of WebText samples
# as prompt and the rest as completion.

import json
import os
import numpy as np
import argparse

from transformers import RobertaTokenizer
from tools import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('dataset_file', type=str, help='dataset file name')
parser.add_argument('--prompt_len', type=int, default=100, help='prompt length')
parser.add_argument('--num_comp', type=int, default=20000, help='number of completions to create')

args = parser.parse_args()
dataset_file = args.dataset_file
prompt_len = args.prompt_len
num_comp = args.num_comp

data_dir = os.path.dirname(dataset_file)
filename = os.path.basename(dataset_file)
output_filename = filename[:filename.find('.')] + '_completion' + filename[filename.find('.'):]
output_file = os.path.join(data_dir, output_filename)

# Load dataset
print('Loading dataset %s' % dataset_file)
samples = []
with open(dataset_file, 'r') as f:
    for line in f:
        samples.append(json.loads(line)['text'])

print('Loaded %d samples' % len(samples))

# Shuffle dataset and select max_samples
np.random.shuffle(samples)

# Create tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create prompts and completions
json_list = []

for i in range(len(samples)):
    if len(json_list) >= num_comp:
        break
    tokens = tokenizer.tokenize(samples[i])
    if len(tokens) >= prompt_len:
        prompt = tokenizer.convert_tokens_to_string(tokens[:prompt_len])
        completion = tokenizer.convert_tokens_to_string(tokens[prompt_len:])
        json_list.append({'prompt': prompt.strip(), 'completion': completion.strip()})

    # Print progress
    done = len(json_list) / num_comp
    print('    Processing ...  %s' % progress_bar(done), end='\r')

print('\nCreated %d prompts and completions' % len(json_list))

# Save prompts and completions
with open(output_file, 'w') as f:
    for json_dict in json_list:
        f.write(json.dumps(json_dict) + '\n')

print('Saved prompts and completions to %s' % output_file)
