# Empirical evaluation of the impossibility result using the OpenAI detector
# based on the RoBERTa model. We use the WebText dataset as human-generated
# text, and eight GPT-2 output datasets as machine-generated text.

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score

# Hyperparameters
batch_size = 100
seq_len = 25

# List of datasets to evaluate
datasets = [
    'webtext',      # Human-generated text
    'small-117M',  'small-117M-k40',    # Machine-generated text
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

# Load OpenAI detector
print("Loading OpenAI detector")
detector = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
detector_tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
detector.eval()

# Set PyTorch device to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)
detector.to(device)

# Evaluate OpenAI detector on WebText dataset
print("Evaluating OpenAI detector on WebText dataset")
webtext_scores = torch.empty(0)

for i in range(0, len(webtext), batch_size):
    # print("Batch %d" % (i // batch_size))
    batch = webtext[i:i+batch_size]
    
    # Collect scores
    inputs = detector_tokenizer(batch, return_tensors="pt", max_length=seq_len, truncation=True, padding=True)
    inputs.to(device)
    outputs = detector(**inputs)
    logits = outputs[0]
    scores = torch.softmax(logits, dim=1)[:,1].tolist()
    webtext_scores = torch.cat((webtext_scores, torch.tensor(scores)))

# Evaluate OpenAI detector on GPT-2 datasets
print("Evaluating OpenAI detector on GPT-2 datasets")
gpt2_scores = {}
for ds in datasets:
    if ds == 'webtext':
        continue
    
    print("Dataset: %s" % ds)
    gpt2_scores[ds] = torch.empty(0)

    for i in range(0, len(gpt2[ds]), batch_size):
        # print("Batch %d" % (i // batch_size))
        batch = gpt2[ds][i:i+batch_size]

        # Collect scores
        inputs = detector_tokenizer(batch, return_tensors="pt", max_length=seq_len, truncation=True, padding=True)
        inputs.to(device)
        outputs = detector(**inputs)
        logits = outputs[0]
        scores = torch.softmax(logits, dim=1)[:,1].tolist()
        gpt2_scores[ds] = torch.cat((gpt2_scores[ds], torch.tensor(scores)))


# Calculate AUROC from scores
print("Calculating AUROC")
auroc = {}
for ds in datasets:
    if ds == 'webtext':
        continue
    
    y_true = torch.cat((torch.ones(gpt2_scores[ds].shape[0]), torch.zeros(webtext_scores.shape[0])))
    y_pred = torch.cat((webtext_scores, gpt2_scores[ds]))
    auroc[ds] = roc_auc_score(y_true, y_pred)

# Update AUROC in JSON file
json_file = 'auroc.json'
auroc_json = {}
print("Saving AUROC to %s" % json_file)
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        auroc_json = json.load(f)

auroc_json['seq_len_' + str(seq_len)] = auroc

with open(json_file, 'w') as f:
    json.dump(auroc_json, f, indent=2)
