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
from utils import load_model, parse_args, generate
from datasets import load_dataset
from six.moves import cPickle as pkl
import nltk
nltk.download('punkt')

args = parse_args()
dataset = load_dataset("xsum")
dataset = dataset["test"]["document"][:500]
IND = []

s1 = []
s2 = []

for i in range(len(dataset)):
    sent = nltk.sent_tokenize(dataset[i])
    if len(sent) >= 5:
        sent = nltk.sent_tokenize(dataset[i])[:5]
        s1.append(" ".join(sent))
        if len(s1) == 100:
            break
    if i == 0:
        print(s1[0])

with open(os.path.join(args.DUMP, "Prompt.pkl"), "wb") as f:
    pkl.dump(s1, f)

model, tokenizer, device = load_model(args)
o1, o2 = [], []

for i in range(len(s1)):
    _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ \
        = generate(s1[i], args, model=model, device=device, tokenizer=tokenizer)
    o1.append(decoded_output_without_watermark)
    o2.append(decoded_output_with_watermark)
    print(i+1, '/', len(s1))

with open(os.path.join(args.DUMP, "Llm_output.pkl"), "wb") as f:
    pkl.dump(o1, f)

with open(os.path.join(args.DUMP, "Llm_watermarked.pkl"), "wb") as f:
    pkl.dump(o2, f)