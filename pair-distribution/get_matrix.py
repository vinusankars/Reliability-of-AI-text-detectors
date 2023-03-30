from wordlist import *
import numpy as np
import os
import sys 
sys.path.append("../lm-watermarking/")
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
from utils import load_model, parse_args, generate
from datasets import load_dataset
from six.moves import cPickle as pkl

N = len(words)
M = np.zeros((N, N))
args = parse_args()
model, tokenizer, device = load_model(args)
found = 0
thresh = 5000

while True:

    input_text = " ".join([words[i] for i in np.random.randint(0, len(words), size=100)])
    _, _, _, decoded_output_with_watermark, _ = generate(input_text, args, model=model, device=device, tokenizer=tokenizer)
    tokens = decoded_output_with_watermark.split(" ")

    for j in range(len(tokens)-1):
        if tokens[j].lower() in words and tokens[j+1].lower() in words:
            i1 = word_to_index[tokens[j].lower()]
            i2 = word_to_index[tokens[j+1].lower()]
            M[i1][i2] += 1
            found += 1
        print("{:8d} pairs found.".format(found), flush=True, end="\r")

    if found >= 10**6:
        np.save(os.path.join(args.DUMP, "matrix"), M)
        break 
        
    if found  >= thresh:
        np.save(os.path.join(args.DUMP, "matrix"), M)
        thresh += 5000