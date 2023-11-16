# Plot the TV values for GPT-2 and GPT-3 from tv_gpt.json as a bar chart.

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tv_file_path', type=str, help='path to TV json file')
args = parser.parse_args()

tv_file_path = args.tv_file_path
# plot_file_path = tv_file_path.replace('.json', '.png')
plot_file_path = tv_file_path + 'tv.png'
print('Plotting TV values to %s' % plot_file_path)

data_list = []

for i in range(1, 7):
    # Input JSON data
    data = json.load(open(tv_file_path + 'tv_run_' + str(i) + '.json', 'r'))

    # Preprocess the data to create a DataFrame
    data_list += [
        {'Model': model, 'Sequence Length': seq_len, 'Value': value}
        for seq_len, values in data.items()
        for model, value in values.items()
    ]

df = pd.DataFrame(data_list)

# Plot the bar chart using seaborn
sns.set(style='darkgrid')
plt.figure()
sns.barplot(x='Sequence Length', y='Value', hue='Model',
            data=df, palette='colorblind') # palette='deep'
# plt.title('TV w.r.t WebText: GPT-2 vs GPT-3')
plt.xlabel('Sequence Length', fontsize=18)
plt.ylabel('Total Variation Estimate', fontsize=18)
# plt.ylim([0, 1])
plt.legend(loc='upper left', # bbox_to_anchor=(0.5, 1.1),
            ncol=2,
           fontsize=12,
           framealpha=1,
           fancybox=True
           )
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.18, left=0.18, top=0.85)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),
#           ncol=2,
#           # fancybox=True, shadow=True,
#           # fontsize='small'
#           )

# Save plot to file
plt.savefig(plot_file_path)
print('Saved plot to %s' % plot_file_path)
