# Compare the AUROC for the best-possible detector to
# the baseline AUROC (random classifier).

import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

# AUROC upper bound
xpoints1 = np.arange(0, 100.00001, 1)
ypoints1 = 100 - (xpoints1 ** 2 / 200)      # 0.5 + xpoints1 - (xpoints1 ** 2 / 2)
plt.plot(xpoints1, ypoints1, label='Best Possible')
plt.text(40.5, 78, 'Best Detector', fontsize=14)

# Baseline
xpoints2 = np.array([0, 100])
ypoints2 = np.array([50, 50])
plt.plot(xpoints2, ypoints2, 'k--', label='Random Classifier')
plt.text(37, 45, 'Random Classifier', fontsize=14)

plt.xlim([0, 100])
plt.xticks(np.arange(0, 100.0001, 10))
plt.ylim([0, 100.2])
plt.yticks(np.arange(0, 100.0001, 10))
plt.grid(linewidth=1.5)
plt.xlabel('Overlap between AI and Human Text', fontsize=15)
plt.ylabel('Detection Performance (AUROC)', fontsize=15)
# plt.legend(frameon='True', fontsize=12)

# Save the plot
if not os.path.exists('plots'):
    os.makedirs('plots')
plt.savefig('plots/perfromance_vs_overlap.png')
