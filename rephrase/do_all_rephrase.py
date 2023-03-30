from parrot import Parrot
import os
import numpy as np
import sys 
sys.path.append("../lm-watermarking/")
import torch
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

for knob in [.0, .04, .08, .16, .25]:
    R = []
    start = time()

    for i in range(0, len(o2), 1):
        phrases = nltk.sent_tokenize(o2[i])
        r = []
        for ii, phrase in enumerate(phrases):
            para_phrases = []
            para_phrases = parrot.augment(input_phrase=phrase, 
                            diversity_ranker="levenshtein",
                            do_diverse=False, 
                            max_return_phrases = 10, 
                            max_length=500, 
                            adequacy_threshold = 1.-knob, 
                            fluency_threshold = 1.-knob,
                            use_gpu=True)

            if para_phrases == None:
                para_phrases = [[phrases[ii]]]

            r.append(para_phrases[0][0])

        R.append(". ".join(r))
        print("Knob:{:.2f}, {:3d}/{:3d} Time:{:.2f}".format(knob, i+1, len(o2), (time()-start)/60), flush=True, end="\r")


    with open(os.path.join(args.DUMP, "rephrased_{:.2f}.pkl".format(knob)), "wb") as f:
        pkl.dump(R, f)
    print()