import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from six.moves import cPickle as pkl
import os


DUMP = "/SCRATCH/DIR/"
with open(os.path.join(DUMP, "watermarked_text.pkl"), "rb") as f:
    prompts = pkl.load(f)
# watermarked_text.pkl contains a list of strings.
# Each string is a watermarked LLM passage completion
# generated using a watermarked OPT-1.3B model on the XSum dataset

CHECKPOINT = "/SCRATCH/DIR/models--kalpeshk2011--dipper-paraphraser-xxl/snapshots/8c2c4ba919079d8ed1609ae5b5f0e14a08ed2d54/"
# path to DIPPER checkpoint
# Download. See: https://github.com/martiansideofthemoon/ai-detection-paraphrases


# Reference: https://github.com/martiansideofthemoon/ai-detection-paraphrases/blob/main/dipper_paraphrases/paraphrase_minimal.py
class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')

        with init_empty_weights():
            self.model = T5ForConditionalGeneration.from_pretrained(model)

        self.model.tie_weights()

        self.model.model_parallel = True
        self.model = load_checkpoint_and_dispatch(self.model, CHECKPOINT, device_map="auto", no_split_module_classes=["encoder", "decoder", "lm_head", "shared"])

        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()
        self.model.tie_weights()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text
    

if __name__ == "__main__":

    P = 5 # number of rounds of paraphrasing
    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")
    out = [[] for j in range(P)]

    for i in range(len(prompts)):
        input_text = prompts[i]
        prompt = "" # this goes as context/prefix to paraphrase in 1st round

        for j in range(P):

            if j == P-1:
                order_diversity = 60
            else:
                order_diversity = 0

            output = dp.paraphrase(input_text, lex_diversity=60, order_diversity=order_diversity, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=750)
            out[j].append(output)

            if i == 0:
                print("\n\nInput:", input_text)
                print("\n\nOutput:", j, output)

            prompt = input_text # use current input as prefix for next round
            input_text = output # recursive paraphrase. current output goes as input in next round.

            print("{:2d}/{:2d}, {:2d}/{:2d}".format(i+1, len(prompts), j+1, P), end="\r", flush=True)

    for j in range(P):
        # save paraphrased outputs
        with open(os.path.join(DUMP, "watermark_dipper_{}.pkl".format(j)), "wb") as f:
            pkl.dump(out[j], f)
