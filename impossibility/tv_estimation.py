# Python script to estimate the total variation distance between WebText and GPT-2
# output distributions using semantic features from a RoBERTa-based text classifier.

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression

# Hyperparameters
batch_size = 100
seq_len = 25

# Function to estimate total variation using Linear classification on the features
def tv_estimate(webtext_features, gpt2_features):
    # Concatenate the features
    X = torch.cat((webtext_features, gpt2_features), dim=0)
    y = torch.cat((torch.ones(webtext_features.shape[0]), torch.zeros(gpt2_features.shape[0])))

    # Scale the features
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    # Shuffle the data
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    y = y[perm]

    # Split the data into train and test
    split = int(0.75 * X.shape[0])
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # Train a linear classifier
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = clf.predict_proba(X_test)[:,1]

    # Threshold the predictions
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    # Find true positives, false positives, true negatives, false negatives
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_test[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            fn += 1

    # Find rates
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)

    return max(abs(tpr - fpr), abs(tnr - fnr))

# List of datasets to evaluate
datasets = [
    'webtext',      # Human-generated text
    'small-117M',  'small-117M-k40',    # Machine-generated text from GPT-2
    'medium-345M', 'medium-345M-k40',
    'large-762M',  'large-762M-k40',
    'xl-1542M',    'xl-1542M-k40',
]

# Download datasets into data directory
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

url_prefix = 'https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/'

for ds in datasets:
    filename = ds + '.test.jsonl'
    if not os.path.exists(os.path.join(data_dir, filename)):
        print("Downloading dataset: %s" % ds)
        os.system('wget ' + url_prefix + filename + ' -P ' + data_dir)

# Download GPT-3 output dataset
url_prefix = 'https://github.com/openai/gpt-3/raw/master/'
filename = '175b_samples.jsonl'

if not os.path.exists(os.path.join(data_dir, filename)):
    print("Downloading dataset: %s" % ds)
    os.system('wget ' + url_prefix + filename + ' -P ' + data_dir)

# Load datasets into memory from data directory
gpt2 = {}
for ds in datasets:
    filename = ds + '.test.jsonl'
    if ds == 'webtext':
        webtext = []
        with open(os.path.join(data_dir, filename), 'r') as f:
            for line in f:
                webtext.append(json.loads(line)['text'])
            
        print('Loaded dataset %s with %d samples' % (ds, len(webtext)))
    else:
        gpt2[ds] = []
        with open(os.path.join(data_dir, filename), 'r') as f:
            for line in f:
                gpt2[ds].append(json.loads(line)['text'])
        
        print('Loaded dataset %s with %d samples' % (ds, len(gpt2[ds])))

# Load GPT-3 dataset
gpt3 = []
with open(os.path.join(data_dir, '175b_samples.jsonl'), 'r') as f:
    for line in f:
        gpt3.append(json.loads(line))

print('Loaded dataset %s with %d samples' % ('gpt3', len(gpt3)))

# Load semantic features model
print("Loading semantic features model")
model_name = 'roberta-base-openai-detector'     # 'roberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Set PyTorch device to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)
model.to(device)

num_features = model.classifier.dense.weight.shape[1]

# Add a hook to the dense linear layer of the classifier to get feature vectors
activations = []
def hook(module, input, output):
    activations.append(output.detach().cpu())

model.classifier.dense.register_forward_hook(hook)

# Evaluate model on WebText and GPT-2 datasets and compute total variation
tv = {}
for ds in datasets:
    if ds == 'webtext':
        # Evaluate model on WebText
        print("Evaluating model on WebText")
        webtext_features = torch.zeros((len(webtext), num_features))
        for i in range(0, len(webtext), batch_size):
            batch = webtext[i:i+batch_size]
            batch = tokenizer(batch, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch)
            webtext_features[i:i+batch_size] = activations[0]
            activations = []
                    
    else:
        # Evaluate model on GPT-2
        print("Evaluating model on GPT-2 dataset: %s" % ds)
        gpt2_features = torch.zeros((len(gpt2[ds]), num_features))
        for i in range(0, len(gpt2[ds]), batch_size):
            batch = gpt2[ds][i:i+batch_size]
            batch = tokenizer(batch, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch)
            gpt2_features[i:i+batch_size] = activations[0]
            activations = []

        # Estimate total variation
        print("Estimating total variation between WebText and GPT-2 dataset: %s" % ds)
        tv[ds] = tv_estimate(webtext_features, gpt2_features)
        print("Total variation between WebText and GPT-2 dataset: %s is: %f" % (ds, tv[ds]))

# Evaluate model on GPT-3 dataset and compute total variation
print("Evaluating model on GPT-3 dataset")
gpt3_features = torch.zeros((len(gpt3), num_features))
for i in range(0, len(gpt3), batch_size):
    batch = gpt3[i:i+batch_size]
    batch = tokenizer(batch, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        model(**batch)
    gpt3_features[i:i+batch_size] = activations[0]
    activations = []

# Estimate total variation
print("Estimating total variation between WebText and GPT-3 dataset")
tv['gpt3'] = tv_estimate(webtext_features, gpt3_features)
print("Total variation between WebText and GPT-3 dataset is: %f" % tv['gpt3'])

# Update tv estimates in JSON file
json_file = 'tv_estimates.json'
tv_estimates = {}
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        tv_estimates = json.load(f)

tv_estimates['seq_len_' + str(seq_len)] = tv

with open(json_file, 'w') as f:
    json.dump(tv_estimates, f, indent=2)
