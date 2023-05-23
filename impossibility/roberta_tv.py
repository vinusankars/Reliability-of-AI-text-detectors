# Estimating the TV between human and AI-generated text using RoBERTa.
#   Step 1: Train a classifier to classify human and AI-generated text.
#   Step 2: Use the trained model to calculate the best TV between
#           human and AI-generated text on the validation datasets by
#           thresholding the softmax scores.
#   Step 3: Save the model and the threshold for best TV.

import argparse
import json
import os
import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import roc_curve, auc

from tools import progress_bar, dataset_len, load_texts

parser = argparse.ArgumentParser()
parser.add_argument('human_text', type=str, help='human text dataset name')
parser.add_argument('ai_text', type=str, help='AI-generated text dataset name')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--seq_len', type=int, default=50, help='sequence length')
parser.add_argument('--tv_model', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large'], help='TV model name')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')

args = parser.parse_args()

# Settings
train_samples = 10000
valid_samples = 4000
num_epochs = 10

data_dir = args.data_dir
sub_data_dir = os.path.join(*(data_dir.split('/')[1:]))
human_text = args.human_text
ai_text = args.ai_text
batch_size = args.batch_size
seq_len = args.seq_len
tv_model = args.tv_model

print('Human text:\t%s' % human_text)
print('AI text:\t%s' % ai_text)
print('Data dir:\t%s' % data_dir)
print('Batch size:\t%d' % batch_size)
print('Sequence len:\t%d' % seq_len)
print('TV model:\t%s' % tv_model)

# Create model directory
model_dir = os.path.join('models', sub_data_dir,
                         human_text, ai_text,
                         'seq_len_' + str(seq_len), tv_model)
print('Model dir:\t%s' % model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load human text dataset from data directory
filename = human_text + '.train.jsonl'
human_text_train = load_texts(os.path.join(data_dir, filename), human_text)     
np.random.shuffle(human_text_train)
print('Loaded %d samples from %s' % (len(human_text_train), filename))
human_text_train = dataset_len(human_text_train, train_samples)

filename = human_text + '.valid.jsonl'
human_text_valid = load_texts(os.path.join(data_dir, filename), human_text)
np.random.shuffle(human_text_valid)
human_text_valid = human_text_valid[:valid_samples]
print('Loaded %d samples from %s' % (len(human_text_valid), filename))

# Load AI-text dataset from data directory
filename = ai_text + '.train.jsonl'
ai_text_train = load_texts(os.path.join(data_dir, filename), ai_text)
np.random.shuffle(ai_text_train)
print('Loaded %d samples from %s' % (len(ai_text_train), filename))
ai_text_train = dataset_len(ai_text_train, train_samples)

filename = ai_text + '.valid.jsonl'
ai_text_valid = load_texts(os.path.join(data_dir, filename), ai_text)
np.random.shuffle(ai_text_valid)
ai_text_valid = ai_text_valid[:valid_samples]
print('Loaded %d samples from %s' % (len(ai_text_valid), filename))

# Load RoBERTa classifier
tokenizer = RobertaTokenizer.from_pretrained(tv_model)
model = RobertaForSequenceClassification.from_pretrained(tv_model, num_labels=2)

# Set PyTorch device to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)
model.to(device)

# Create training data
train_texts = human_text_train + ai_text_train
print('Training on %d samples' % len(train_texts))
train_labels = [0] * len(human_text_train) + [1] * len(ai_text_train)

# Shuffle training data
idx = torch.randperm(len(train_texts))
train_texts = [train_texts[i] for i in idx]
train_labels = [train_labels[i] for i in idx]

# Create validation data
valid_texts = human_text_valid + ai_text_valid
print('Validating on %d samples' % len(valid_texts))
valid_labels = [0] * len(human_text_valid) + [1] * len(ai_text_valid)

# Train classifier
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

best_tv = 0
avg_acc = 0
avg_loss = 0

logfile = open(os.path.join(model_dir, 'log.txt'), 'w')
logfile.write('Epoch\tTV\tthreshold\n')

for epoch in range(num_epochs):
    print('Epoch %d/%d' % (epoch + 1, num_epochs))
    model.train()
    print('Training...')
    for i in range(0, len(train_texts), batch_size):
        # Prepare batch
        batch_texts = train_texts[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_labels = torch.tensor(batch_labels).to(device)

        # Forward pass
        outputs = model(**batch_inputs, labels=batch_labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == batch_labels).item() / len(batch_labels)

        # Update running average
        batch_num = i / batch_size + 1
        avg_acc = (avg_acc * (batch_num - 1) + acc) / batch_num
        avg_loss = (avg_loss * (batch_num - 1) + loss.item()) / batch_num

        # Print progress
        progress = (i + batch_size) / len(train_texts)
        batch_info = '  Batch %3d/%d, loss: %.4f, acc: %.4f' % (batch_num, len(train_texts)/batch_size, avg_loss, avg_acc)
        print(batch_info + ' ' * 4 + progress_bar(progress), end='\r')
        info_len = len(batch_info)

    print('\nFinished training epoch')

    # Evaluate on validation set
    scores = torch.zeros(len(valid_texts))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]
            batch_labels = valid_labels[i:i+batch_size]
            batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = torch.tensor(batch_labels).to(device)

            outputs = model(**batch_inputs, labels=batch_labels)
            logits = outputs.logits

            scores_batch = torch.softmax(logits, dim=1)[:, 1]
            scores[i:i+batch_size] = scores_batch

            # Print progress
            progress = (i + batch_size) / len(valid_texts)
            msg = 'Evaluating on validation set...'
            print(msg + ' ' * (info_len - len(msg) + 4) + progress_bar(progress), end='\r')

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(valid_labels, scores)
    roc_auc = auc(fpr, tpr)
    tv = np.amax(tpr - fpr)
    threshold = thresholds[np.argmax(tpr - fpr)]
    print('\nAUC: %.4f, TV: %.4f (%.4f)\nthreshold = %.4f' % (roc_auc, tv, max(tv, best_tv), threshold))
    logfile.write('%d\t%.4f\t%.4f\n' % (epoch + 1, tv, threshold))
    logfile.flush()

    # Save best model
    if tv > best_tv:
        best_tv = tv
        print('Saving best model...')
        torch.save(model.state_dict(), os.path.join(model_dir, 'state.pt'))

        f = open(os.path.join(model_dir, 'threshold'), 'w')
        f.write(str(threshold))
        f.close()

logfile.close()