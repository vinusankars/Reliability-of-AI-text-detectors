import numpy as np
import json

DATASETS = [
        'webtext',      # Human-generated text
        'small-117M',  'small-117M-k40',    # GPT-2 generated text
        'medium-345M', 'medium-345M-k40',
        'large-762M',  'large-762M-k40',
        'xl-1542M',    'xl-1542M-k40'
    ]

def load_texts(filepath, dataset_name):
    # Load text dataset from filepath
    texts = []
    with open(filepath, 'r') as f:
        for line in f:
            if dataset_name in DATASETS:
                texts.append(json.loads(line)['text'].strip())
            elif '_completion' in dataset_name:
                texts.append(json.loads(line)['completion'].strip())
            elif '_paraphrased' in dataset_name:
                texts.append(json.loads(line)['paraphrase'].strip())
            elif '_10k' in dataset_name:
                texts.append(json.loads(line)['text'].strip())
            else:
                texts.append(json.loads(line).strip())

    return texts

def dataset_len(dataset, num_samples):
    if num_samples < len(dataset):
        # Truncate dataset
        dataset = dataset[:num_samples]
        print('Truncated to %d samples' % len(dataset))

    elif num_samples > len(dataset):
        # Extend dataset
        idx = np.random.randint(len(dataset), size=num_samples - len(dataset))
        dataset.extend([dataset[i] for i in idx])
        print('Extended to %d samples' % len(dataset))

    return dataset

def progress_bar(done, done_symbol='█', left_symbol='▒', length=25):
    bar_done = int(done * length)
    bar_left = length - bar_done
    return done_symbol * bar_done + left_symbol * bar_left + ' %3d%%' % (done * 100)

def calculate_tv(scores, labels, threshold):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if scores[i] >= threshold:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == 1:
                fn += 1
            else:
                tn += 1

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return tpr - fpr


# Test cases
if __name__ == '__main__':
    # Test text loading
    dataset_name = 'webtext_completion'
    # dataset_name = 'webtext_completion'
    texts = load_texts('data/completions/' + dataset_name + '.test.jsonl', dataset_name)
    print('Loaded %d texts' % len(texts))
    print(texts[0])
    exit()

    from time import sleep

    # Test progress bar
    for i in np.arange(0, 1.01, 0.01):
        print(' ' * 4 + progress_bar(i), end='\r')
        sleep(0.1)

    # Test calculate_tv
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    threshold = 0.5
    tv = calculate_tv(scores, labels, threshold)
    print('\nTV: %.3f' % tv)
