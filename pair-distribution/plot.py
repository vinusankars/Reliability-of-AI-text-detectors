from wordlist import *
import numpy as np
import os
import sys
sys.path.append("../lm-watermarking/")
from six.moves import cPickle as pkl
from matplotlib import pyplot as plt
from utils import parse_args
import matplotlib
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set_theme()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

args = parse_args()
M = np.load(os.path.join(args.DUMP, "matrix.npy"))

W = "the"
i1 = word_to_index[W]
x = np.stack([index_to_word[i] for i in range(181)])
y = M[i1]/M[i1].sum()
order = np.argsort(y)[-50:][::-1]
x = x[order]
y = y[order]

# plt.figure(figsize=(12,3), dpi=500)
# plt.plot(x, y, '-+', markersize=4)
# plt.xlabel("Top 50 words")
# plt.ylabel("Green list\nscore")
# # plt.title("Suffix score for the word '{}'".format(W))
# plt.xticks(rotation=90)
# # plt.yticks(np.round(100*np.linspace(0, y.max(), 5))/100)
# plt.semilogy()
# plt.tight_layout()
# plt.savefig("images/{}.png".format(W))

plt.figure(figsize=(7,6), dpi=500)
ticks = [index_to_word[i] for i in range(20)]
M = M[:20, :20]
M1 = np.stack([M[i]/M[i].sum() for i in range(len(M))])
ax = sns.heatmap(M1, linewidth=0.05, xticklabels=ticks, yticklabels=ticks, norm=LogNorm())
plt.tight_layout()
plt.savefig("images/heat.png")