# Evaluate the TV estimator models on WebText and GPT-3 test datasets

import os
import json
import numpy as np
import torch
import argparse

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tools import progress_bar, calculate_tv, load_texts

parser = argparse.ArgumentParser()
parser.add_argument('--shorts', action='store_true', help='evaluate on shorts datasets')
parser.add_argument('--model_name', type=str, default='roberta-base', help='name of model to use')
args = parser.parse_args()

# Settings
batch_size = 100
data_dir = 'data'
shorts = args.shorts
model_name = args.model_name
ds_tag = {
    'gpt2': 'GPT-2',
    'gpt3': 'GPT-3',
    'chatgpt': 'ChatGPT',
    'small-117M': 'GPT-2-S',
    'medium-345M': 'GPT-2-M',
    'large-762M': 'GPT-2-L',
    'xl-1542M': 'GPT-2-XL'
}

# GPT datasets to evaluate on
# GPT_DS = ['gpt2', 'gpt3']
# GPT_DS = ['chatgpt']
GPT_DS = ['small-117M', 'medium-345M', 'large-762M', 'xl-1542M']


# Load RoBERTa classifier
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load Webtext test dataset from data directory
dataset_name = 'webtext_shorts' if shorts else 'webtext'
filename = dataset_name + '.test.jsonl'
webtext_test = load_texts(os.path.join(data_dir, filename), dataset_name)

print('Loaded dataset %s with %d samples' % (filename, len(webtext_test)))

# Get TV estimate from json file
json_file = ('tv_shorts_' if shorts else 'tv_') + model_name + '_' + GPT_DS[0] + '.json'
tv_json = {}
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        tv_json = json.load(f)

for gpt_ds in GPT_DS:
    # Load GPT test dataset from data directory
    dataset_name = gpt_ds + ('_shorts' if shorts else '')
    filename = dataset_name + '.test.jsonl'
    gpt_test = load_texts(os.path.join(data_dir, filename), dataset_name)

    print('Loaded dataset %s with %d samples' % (filename, len(gpt_test)))

    tv_vals = {}
    for seq_len in [25, 50, 75, 100]:
        print('Evaluating for seq_len = %d' % seq_len)
        model_dir = 'models/'+ ('webtext_shorts/' if shorts else 'webtext/')
        model_dir += gpt_ds + ('_shorts' if shorts else '') + '/seq_len_' + str(seq_len) + '/' + model_name
        
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
with open(json_file, 'w') as f:
    json.dump(tv_json, f, indent=2)
