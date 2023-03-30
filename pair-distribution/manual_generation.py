from wordlist import *
import numpy as np
import os
import sys
sys.path.append("../lm-watermarking/")
from utils import parse_args

args = parse_args()
M = np.load(os.path.join(args.DUMP, "matrix.npy"))

def wlist(w):

    if w == None or w not in word_to_index:
        print("**Word NOT in list.**")
        ind = np.arange(len(M))
        np.random.shuffle(ind)
        return np.stack(words)[ind]

    else:
        print("**Word in list.**")

    ind = word_to_index[w]
    score = M[ind]
    W = np.stack([index_to_word[i] for i in range(len(M))])
    order = np.argsort(score)[::-1]
    score, W = score[order], W[order]
    W = W[score > 0]

    return W

Instruction = ("\n\nAt every step you will be provided a list of words.\n"
                "These words are arranged in decreasing order of green list score.\n"
                "If you choose the first word, higher your chances of getting detected by watermarker.\n"
                "Add word by word. At every step, you will be shown the sentence you have composed so far.\n"
                "You can add out-of-vocbulary words some times to build a meaningful sentence.\n"
                "It's better if you avoid punctuations in your sentences.\n"
                "Type all lower case, simply words without typos.\n"
                "Type '!!!' to stop the generation and '-del' to remove last word.\n\n")

print(Instruction)
print("Input the passage you want to start with below (add space in the end). If no passage available, press enter.")
passage = input()
if passage == "":
    w = None
else:
    if passage[-1].endswith("."):
        w = passage.split()[-1][:-1]
    else:
        w = passage.split()[-1]
counter = 0

while True:

    print("Word list: {}\n".format(wlist(w)))
    w = input("\nEnter word: ").lower()
    counter += 1

    if w == "!!!":
        passage = passage[:-1]
        break 

    if w == "-del":
        passage = " ".join(passage.split(" ")[:-2]) + " "
        print("\nYour sentence so far ({} words): {}".format(counter, passage[:-1]))
        w = passage.split(" ")[-2]
        continue
    
    passage += w + " "
    print("\nYour sentence so far ({} words): {}".format(counter, passage[:-1]))
    print("\n", "#"*30, "\n\n")

print("\n", "#"*30, "\n\n")
print("\n\nYour final sentence is ({} words): {}".format(counter, passage))