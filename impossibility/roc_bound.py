# Compare the AUROC for the best-possible detector to
# the baseline AUROC (random classifier).

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style='darkgrid')

# AUROC upper bound
xpoints1 = np.arange(0, 1.0001, 0.01)
ypoints1 = 0.5 + xpoints1 - (xpoints1 ** 2 / 2)
plt.plot(xpoints1, ypoints1, label='Best Possible')
plt.text(0.33, 0.71, 'Best Possible', fontsize=18)

# Baseline
xpoints2 = np.array([0, 1])
ypoints2 = np.array([0.5, 0.5])
plt.plot(xpoints2, ypoints2, 'k--', label='Random Classifier')
plt.text(0.26, 0.43, 'Random Classifier', fontsize=18)

plt.xlim([0, 1])
plt.xticks(np.arange(0, 1.0001, 0.1), fontsize=18)
plt.ylim([0, 1.005])
plt.yticks(np.arange(0, 1.0001, 0.1), fontsize=18)
plt.grid(linewidth=1.5)
plt.xlabel('Total Variation', fontsize=18)
plt.ylabel('AUROC', fontsize=18)
# plt.legend(frameon='True', fontsize=12)
plt.subplots_adjust(bottom=0.15, left=0.15)

# Save the plot
if not os.path.exists('results'):
    os.makedirs('results')
plt.savefig('results/roc_bound.png')
