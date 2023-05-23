# Evaluate the TV estimator models on WebText and GPT-3 test datasets

import os
import json
import numpy as np
import torch
import argparse

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tools import progress_bar, calculate_tv, load_texts

parser = argparse.ArgumentParser()
parser.add_argument('human_text', type=str, help='human text dataset')
parser.add_argument('--model_name', type=str, default='roberta-base', help='name of model to use')
parser.add_argument('--data_dir', type=str, default='data', help='name of data directory')
parser.add_argument('--tag', type=str, default='', help='tag for saving results')
args = parser.parse_args()

# Settings
humantext_dataset_name = args.human_text
batch_size = 100
data_dir = args.data_dir
sub_data_dir = os.path.join(*(data_dir.split('/')[1:]))
shorts = args.shorts
model_name = args.model_name
tag = args.tag
ds_tag = {
    'small-117M': 'GPT-2-S',
    'medium-345M': 'GPT-2-M',
    'large-762M': 'GPT-2-L',
    'xl-1542M': 'GPT-2-XL',
    'text-ada-001_completion': 'GPT-3-Ada',
    'text-babbage-001_completion': 'GPT-3-Babbage',
    'text-curie-001_completion': 'GPT-3-Curie'
}

# GPT datasets to evaluate on
GPT_DS = ['small-117M', 'medium-345M', 'large-762M', 'xl-1542M']
# GPT_DS = ['text-ada-001_completion', 'text-babbage-001_completion', 'text-curie-001_completion']

# Load RoBERTa classifier
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load humantext test dataset from data directory
filename = humantext_dataset_name + '.test.jsonl'
webtext_test = load_texts(os.path.join(data_dir, filename), humantext_dataset_name)

print('Loaded dataset %s with %d samples' % (filename, len(webtext_test)))

# Get TV estimate from json file
results_dir = os.path.join('results', ('tv_shorts' if shorts else 'tv'), sub_data_dir, humantext_dataset_name, model_name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

json_file_path = os.path.join(results_dir, GPT_DS[0] + tag + '.json')
print('Loading TV estimate from %s' % json_file_path)
tv_json = {}
if os.path.exists(json_file_path):
    with open(json_file_path, 'r') as f:
        tv_json = json.load(f)

for gpt_ds in GPT_DS:
    # Load GPT test dataset from data directory
    gpt_dataset_name = gpt_ds + ('_shorts' if shorts else '')
    filename = gpt_dataset_name + '.test.jsonl'
    gpt_test = load_texts(os.path.join(data_dir, filename), gpt_dataset_name)

    print('Loaded dataset %s with %d samples' % (filename, len(gpt_test)))

    tv_vals = {}
    for seq_len in [25, 50, 75, 100]:
        print('Evaluating for seq_len = %d' % seq_len)
        model_dir = os.path.join('models', sub_data_dir, humantext_dataset_name)
        model_dir = os.path.join(model_dir, gpt_ds + ('_shorts' if shorts else ''), 'seq_len_' + str(seq_len), model_name)
        
        # Load state dict
        print('Loading model from %s' % model_dir)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'state.pt')))
        model.eval()

        # Load best TV threshold
        with open(os.path.join(model_dir, 'threshold'), 'r') as f:
            threshold = float(f.read())

        # Set PyTorch device to CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: %s" % device)
        model.to(device)

        # Create test data
        test_data = webtext_test + gpt_test
        print('Test data size: %d' % len(test_data))
        test_labels = [0] * len(webtext_test) + [1] * len(gpt_test)

        # Evaluate model on test data
        scores = torch.zeros(len(test_data))

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_texts = test_data[i:i+batch_size]
                batch_labels = test_labels[i:i+batch_size]
                batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                batch_labels = torch.tensor(batch_labels).to(device)

                outputs = model(**batch_inputs)
                logits = outputs.logits

                scores_batch = torch.softmax(logits, dim=1)[:, 1]
                scores[i:i+batch_size] = scores_batch

                # Print progress
                progress = (i + batch_size) / len(test_data)
                print('Evaluating on test data...  ' + progress_bar(progress), end='\r')


        # Calculate TV
        scores = scores.cpu().numpy()
        tv = calculate_tv(scores, test_labels, threshold)
        print('\nTV: %.3f' % tv)
        tv_vals[seq_len] = tv

    # Update TV estimates for GPT dataset
    tv_json[ds_tag[gpt_ds]] = tv_vals


# Save TV values to json file
with open(json_file_path, 'w') as f:
    json.dump(tv_json, f, indent=2)
