# Estimate the total variation distance between Human Text and GPT-2 output
# by estimating the density of the human text and the density of the GPT-2 output.

import json
import numpy as np
import os
import argparse

from tools import progress_bar

from transformers import RobertaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples to use')
parser.add_argument('--num_part', type=int, default=10, help='Number of partitions of the token set')
parser.add_argument('--seq_len', type=int, default=3, help='Sequence length')
parser.add_argument('--tv_file', type=str, default='tv_from_density.json', help='File to store TV values')

args = parser.parse_args()

num_samples = args.num_samples
num_part = args.num_part
seq_len = args.seq_len
tv_file = args.tv_file
ds_tag = {
    'small-117M': 'GPT-2-S',
    'medium-345M': 'GPT-2-M',
    'large-762M': 'GPT-2-L',
    'xl-1542M': 'GPT-2-XL',
    'webtext': 'WT (Ref)',
    'text-ada-001_completion': 'GPT-3-Ada',
    'text-babbage-001_completion': 'GPT-3-Babbage',
    'text-curie-001_completion': 'GPT-3-Curie'
}

print('Number of samples: ' + str(num_samples))
print('Number of partitions: ' + str(num_part))
print('Sequence length: ' + str(seq_len))
print('TV file: ' + tv_file)

tv_json = {}
if os.path.exists(tv_file):
    with open(tv_file, 'r') as f:
        tv_json = json.load(f)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Partition tokens into num_part partitions
def partition_map(tokens, num_part):
    # Map tokens to partitions
    partition = []
    for i in range(len(tokens)):
        partition.append(tokens[i] % num_part)
    return partition

# Get the frequency of each sequence of length seq_len
def seq_freq(dataset_file, seq_len=seq_len, num_part=num_part,
            num_samples=num_samples, tokenizer=tokenizer):

    # Load dataset
    with open(dataset_file, 'r') as f:
        text = []
        for line in f:
            text.append(json.loads(line)['text'])

    # Shuffle dataset
    np.random.shuffle(text)

    print('Loaded ' + str(len(text)) + ' lines from ' + dataset_file)

    # Frequency array
    frequency = np.zeros([num_part] * seq_len)
    # print('Frequency array shape: ' + str(frequency.shape))

    for i in range(num_samples):

        # Tokenize text
        tokens = tokenizer.encode(text[i], max_length=512, truncation=True, padding='max_length')

        # Update density array
        frequency[tuple(partition_map(tokens[1:1+seq_len], num_part))] += 1

        # Print progress
        if (i + 1) % 1000 == 0 or i == 0:
            progress = (i + 1) / num_samples
            print('  Progress: ' + progress_bar(progress), end='\r', flush=True)

    print('')
    return frequency

# WebText sequence frequency array
webtext_frequency = seq_freq('data/webtext.train.jsonl')

# Store TV values
tv_vals = {}

# GPT-2 sequence frequency array
for dataset in ['small-117M', 'medium-345M', 'large-762M', 'xl-1542M']:     # , 'webtext']:
    filename = 'data/'+ dataset + '.train.jsonl'

    gpt2_frequency = seq_freq(filename)

    # Estimate total variation distance
    tv = np.sum(np.abs(webtext_frequency - gpt2_frequency)) / (2 * num_samples)

    tv_vals[ds_tag[dataset]] = tv

    print('Total variation distance between webtext and ' + dataset + ' is ' + str(tv))

tv_json[str(seq_len)] = tv_vals

# Save TV values to file
with open(tv_file, 'w') as f:
    json.dump(tv_json, f, indent=2)