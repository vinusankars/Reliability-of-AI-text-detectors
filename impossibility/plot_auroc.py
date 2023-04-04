# Plot empirical AUROC vs. TV for different datasets
# along with the AUROC upper bound and baseline

import os
import matplotlib.pyplot as plt
import numpy as np
import json

plt.style.use('seaborn')

# List of datasets to plot
datasets = [
    'webtext',      # Human-generated text
    'small-117M',  'small-117M-k40',    # Machine-generated text
    'medium-345M', 'medium-345M-k40',
    'large-762M',  'large-762M-k40',
    'xl-1542M',    'xl-1542M-k40',
]

# Empirical AUROCs from saved results
print("Plotting empirical AUROCs from saved results")
tv_json = {}
auroc_json = {}

for seq_len in [25, 50, 75, 100]:
    print("Plotting for seq_len = %d" % seq_len)

    # AUROC upper bound
    xpoints1 = np.arange(0, 1.00001, 0.01)
    ypoints1 = 0.5 + xpoints1 - (xpoints1 ** 2 / 2)
    plt.plot(xpoints1, ypoints1, label='AUROC Bound')

    # Baseline
    xpoints2 = np.array([0, 1])
    ypoints2 = np.array([0.5, 0.5])
    plt.plot(xpoints2, ypoints2, 'k--', label='Baseline')

    tv_filename = 'tv_from_scores.json'      # 'tv_estimates.json'
    with open(tv_filename, 'r') as f:
        tv_json = json.load(f)

    with open('auroc.json', 'r') as f:
        auroc_json = json.load(f)

    # Plot empirical AUROC vs. TV for GPT-2 datasets
    for ds in datasets:
        if ds == 'webtext':
            continue
        
        mark = None
        color = None

        if 'small' in ds:
            color = 'b'
        elif 'medium' in ds:
            color = 'g'
        elif 'large' in ds:
            color = 'r'
        elif 'xl' in ds:
            color = 'm'

        if 'k40' in ds:
            mark = '^'
        else:
            mark = 'o'

        plt.scatter(tv_json['seq_len_' + str(seq_len)][ds],
                    auroc_json['seq_len_' + str(seq_len)][ds],
                    marker=mark, c=color, alpha=0.6,
                    label='GPT-2: ' + ds)

    # Plot empirical AUROC vs. TV for GPT-3 dataset
    plt.scatter(tv_json['seq_len_' + str(seq_len)]['gpt3'],
                auroc_json['seq_len_' + str(seq_len)]['gpt3'],
                marker='*', c='olive', s=75, alpha=0.8,
                label='GPT-3')

    # Plot formatting
    plt.title('Sequence Length = ' + str(seq_len), fontsize=15)
    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1.0001, 0.1))
    plt.ylim([0, 1.002])
    plt.yticks(np.arange(0, 1.0001, 0.1))
    plt.grid(linewidth=1.5)
    plt.xlabel('Total Variation', fontsize=15)
    plt.ylabel('AUROC', fontsize=15)
    plt.legend(frameon='True', fontsize=9, loc='lower right')

    # Save plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/emp_eval_seq_len_'+ str(seq_len) +'.png')
    print("Saved plot to plots/emp_eval_seq_len_"+ str(seq_len) +".png")
    plt.clf()
