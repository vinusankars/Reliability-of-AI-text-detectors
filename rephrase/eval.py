from six.moves import cPickle as pkl
from utils import parse_args
import os
from evaluate import load
from transformers import AutoTokenizer
import numpy as np

args = parse_args()

def combine(s1, s2):
    if s1[-1] in [" ", "\n"] or s2[0] in [" ", "\n"]:
        return s1 + s2 
    else:
        return s1 + " " + s2

with open(os.path.join(args.DUMP, "prompt.pkl"), "rb") as f:
    P = pkl.load(f)

with open(os.path.join(args.DUMP, "llm_watermrked.pkl"), "rb") as f:
    W = pkl.load(f)

model_name = "facebook/opt-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
perplexity = load("perplexity", module_type="metric")

p_ppl, p_t = [], []
for i in P:
    p_t.append(tokenizer(i, return_tensors="pt").input_ids.shape[1])
for i in range(0, len(P), 10):
    p_ppl.extend(perplexity.compute(predictions=P[i:i+10], model_id=model_name)['perplexities'])


w_ppl, w_t = [], []
wppl = []
C = []
for i in range(len(W)):
    if len(W[i]) == 0:
        continue
    w_t.append(tokenizer(combine(P[i], W[i]), return_tensors="pt").input_ids.shape[1])
    C.append(combine(P[i], W[i]))
for i in range(0, len(C), 10):
    w_ppl.extend(perplexity.compute(predictions=C[i:i+10], model_id=model_name)['perplexities'])

for i in range(len(w_t)):
    wppl.append(np.exp( (np.log(w_ppl[i]) * w_t[i] - np.log(p_ppl[i]) * p_t[i])  / (w_t[i] - p_t[i]) ))
print(w_ppl)
print("watermark ppl: {:.4f}".format(sum(wppl)/len(wppl)))


print("#"*30)
for knob in [.04, .08, .16, .25, .3, .4]:
    print(knob)
    with open(os.path.join(args.DUMP, "rephrased_{:.2f}.pkl".format(knob)), "rb") as f:
        R = pkl.load(f)
    r_ppl, r_t = [], []
    rppl = []
    C = []
    for i in range(len(R)):
        if len(R[i]) == 0:
            continue
        r_t.append(tokenizer(combine(P[i], R[i]), return_tensors="pt").input_ids.shape[1])
        C.append(combine(P[i], R[i]))
    for i in range(0, len(C), 10):
        r_ppl.extend(perplexity.compute(predictions=C[i:i+10], model_id=model_name)['perplexities'])

    for i in range(len(r_t)):
        rppl.append(np.exp( (np.log(r_ppl[i]) * r_t[i] - np.log(p_ppl[i]) * p_t[i]) / (r_t[i] - p_t[i]) ))
    print(rppl)
    print("rephraser ppl: {:.4f}".format(sum(rppl)/len(rppl)))