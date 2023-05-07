# Plot the TV values for GPT-2 and GPT-3 from tv_gpt.json as a bar chart.

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tv_file', type=str, help='name of TV json file')
args = parser.parse_args()

tv_file = args.tv_file
plot_file = os.path.join('plots', tv_file.replace('.json', '.png'))

# Input JSON data
data = json.load(open(tv_file, 'r'))

# Preprocess the data to create a DataFrame
data_list = [
    {'Model': model, 'Sequence Length': seq_len, 'Value': value}
    for model, values in data.items()
    for seq_len, value in values.items()
]

df = pd.DataFrame(data_list)

# Plot the bar chart using seaborn
sns.set(style='darkgrid')
plt.figure()
sns.barplot(x='Sequence Length', y='Value', hue='Model',
            data=df, palette='deep')
# plt.title('TV w.r.t WebText: GPT-2 vs GPT-3')
plt.xlabel('Input Sequence Length')
plt.ylabel('Total Variation Estimate')
plt.ylim([0, 1])
plt.legend(loc='upper left')

# Save plot to file
plt.savefig(plot_file)
print('Saved plot to %s' % plot_file)
