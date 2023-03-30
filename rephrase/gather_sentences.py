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


args = parse_args()
dataset = load_dataset("xsum")
dataset = dataset["test"]["document"][:200]
IND = []

s1 = []
s2 = []

for i in range(len(dataset)):
    if len(dataset[i]) >= 200:
        words = dataset[i].split(" ")
        s1.append(" ".join(words[: 100]))
        s2.append(" ".join(words[100: ]))
        if len(s1) == 100:
            break

with open(os.path.join(args.DUMP, "prompt.pkl"), "wb") as f:
    pkl.dump(s1, f)

with open(os.path.join(args.DUMP, "truth.pkl"), "wb") as f:
    pkl.dump(s2, f)

model, tokenizer, device = load_model(args)
o1, o2 = [], []

for i in range(len(s1)):
    _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ \
        = generate(s1[i], args, model=model, device=device, tokenizer=tokenizer)
    o1.append(decoded_output_without_watermark)
    o2.append(decoded_output_with_watermark)
    print(i+1, '/', len(s1))

with open(os.path.join(args.DUMP, "llm_output.pkl"), "wb") as f:
    pkl.dump(o1, f)

with open(os.path.join(args.DUMP, "llm_watermrked.pkl"), "wb") as f:
    pkl.dump(o2, f)