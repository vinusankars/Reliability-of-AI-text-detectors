# Download human and GPT text datasets to data directory.

import os
import argparse

def main():

    DATASETS = [
        'webtext',      # Human-generated text
        'small-117M',  'small-117M-k40',    # Machine-generated text
        'medium-345M', 'medium-345M-k40',
        'large-762M',  'large-762M-k40',
        'xl-1542M',    'xl-1542M-k40',
    ]

    SPLITS = ['train', 'valid', 'test']

    url_prefix = 'https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/'


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store data')
    parser.add_argument('--ds', type=str, nargs='+', default=DATASETS, choices=DATASETS, help='Webtext or GPT-2 dataset to download')
    parser.add_argument('--split', type=str, nargs='+', default=SPLITS, choices=SPLITS, help='Split of dataset to download')
    args = parser.parse_args()

    datasets = args.ds
    splits = args.split
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for ds in datasets:
        for split in splits:
            filename = '%s.%s.jsonl' % (ds, split)
            if not os.path.exists(os.path.join(data_dir, filename)):
                print("Downloading dataset: %s split: %s" % (ds, split))
                if os.system('wget ' + url_prefix + filename + ' -P ' + data_dir) != 0:
                    print("Error downloading dataset: %s, split: %s" % (ds, split))
                    exit()

            else:
                print("Dataset: %s, split: %s already exists" % (ds, split))

if __name__ == '__main__':
    main()