import os
import sys 
sys.path.append("../../lm-watermarking/")
sys.path.append("../")
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
import numpy 
import torch
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from utils import load_model, parse_args, detect
from datasets import load_dataset
from six.moves import cPickle as pkl

args = parse_args()

model, tokenizer, device = load_model(args)
del model

print("#"*30)
print("Watermarked text stats")

with open(os.path.join(args.DUMP, "Llm_watermarked.pkl"), "rb") as f:
    o2 = pkl.load(f)

total_words, green = 0, 0
watermarked, N = [], 0

for i in range(0, len(o2), 1):
    try:
        result = detect(o2[i], args, device=device, tokenizer=tokenizer)[0]
        total_words += int(result[0][1])
        green += int(result[1][1])
        watermarked.append(int(result[6][1] == "Watermarked"))
        N += 1
    except: 
        pass

print(watermarked)
print("total tokens:", total_words)
print("total green tokens:", green)
print("% green tokens:", green/total_words)
print("total watermarked passages:", sum(watermarked))
print("total passages:", N)
print("acc:", sum(watermarked)/N)

print("#"*30)
print("Rephrased text stats")

with open(os.path.join(args.DUMP, "Rephrase.pkl"), "rb") as f:
    R = pkl.load(f)

for knob in range(5):
    print("\n", knob)
    total_words, green = 0, 0
    watermarked, N = [], 0

    for i in range(0, len(R), 1):
        passage = R[i]
        passage = " ".join([passage[ii][knob] for ii in range(len(passage))])
        if (knob == 0 or knob == 5) and (i == 0 or i == 1):
            print("\n", passage, "\n")
        try:
            result = detect(passage, args, device=device, tokenizer=tokenizer)[0]
            total_words += int(result[0][1])
            green += int(result[1][1])
            watermarked.append(int(result[6][1] == "Watermarked"))
            N += 1
        except:
            pass
        
    print(watermarked)
    print("total tokens:", total_words)
    print("total green tokens:", green)
    print("% green tokens:", green/total_words)
    print("total watermarked passages:", sum(watermarked))
    print("total passages:", N)
    print("acc:", sum(watermarked)/N)
