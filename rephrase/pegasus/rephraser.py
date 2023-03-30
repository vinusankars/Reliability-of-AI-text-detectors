import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os
import numpy as np
import sys 
sys.path.append("../lm-watermarking/")
sys.path.append("../")
from utils import parse_args
from six.moves import cPickle as pkl
from pprint import pprint
from time import time
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

args = parse_args()
os.environ['TRANSFORMERS_CACHE'] = args.TRANSFORMERS_CACHE

with open(os.path.join(args.DUMP, "Llm_watermarked.pkl"), "rb") as f:
    o2 = pkl.load(f)

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

R = []
start = time()

num_beams = 25
num_return_sequences = 25

try:
    for i in range(0, len(o2)):
        print("{:3d}/{:3d} Time:{:.2f}".format(i+1, len(o2), (time()-start)/60), flush=True, end="\r")
        phrases = nltk.sent_tokenize(o2[i])
        r = []
        for ii, phrase in enumerate(phrases):
            output = get_response(phrase, num_return_sequences, num_beams)
            r.append(output[::5])
        R.append(r)
except:
    pass

with open(os.path.join(args.DUMP, "Rephrase.pkl"), "wb") as f:
    pkl.dump(R, f)