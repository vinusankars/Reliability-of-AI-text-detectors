import torch
from parrot import Parrot
import os
import numpy as np
import sys 
sys.path.append("../lm-watermarking/")
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

with open(os.path.join(args.DUMP, "llm_watermrked.pkl"), "rb") as f:
    o2 = pkl.load(f)

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)
R = []
start = time()

try:
    for i in range(0, len(o2)):
        phrases = nltk.sent_tokenize(o2[i])
        r = []
        for ii, phrase in enumerate(phrases):
            para_phrases = []
            j = 0.
            while para_phrases == None or len(para_phrases) < 2: # make sure enough outputs are generated
                para_phrases = parrot.augment(input_phrase=phrase, 
                                diversity_ranker="levenshtein",
                                do_diverse=False, 
                                max_return_phrases = 10, 
                                max_length=200, 
                                adequacy_threshold = 0.9-j, 
                                fluency_threshold = 0.9-j,
                                use_gpu=True)
                j += 0.1
                if j > 0.3:
                    break

            if para_phrases == None:
                continue

            r.append(para_phrases[0][0])

        R.append(". ".join(r))
        print("{:3d}/{:3d} Time:{:.2f}".format(i+1, len(o2), (time()-start)/60), flush=True, end="\r")

except:
    pass

with open(os.path.join(args.DUMP, "rephrased.pkl"), "wb") as f:
    pkl.dump(R, f)