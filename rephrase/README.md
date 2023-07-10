All hyperparameters are set in utils.parse_args()

Change args.TRANSFORMERS_CACHE to PATH where you want the models to be downloaded (somewhere in your scratch memory).
Change args.DUMP to PATH where you want the experiment outputs should be dumped.

Set "export TRANSFORMERS_CACHE="/PATH/TO/CACHE""

Task: using prithivida/parrot_paraphraser_on_T5 to paraphrase watermarked text.

To run:

$ python gather_sentences.py 

$ python rephraser.py

$ python detect.py

$ # TODO: Analyze perplexities of output texts.

Results:

| Text           | LLM watermark | Rephrased |
|----------------|---------------|-----------|
| # tokens       | 17478         | 14147     |
| # green tokens | 10085         | 5681      |
| % detected     | 98            | 57        |

For recursive paraphrasing, we use scripts from Krishna et al. (2023) (Reference: https://github.com/martiansideofthemoon/ai-detection-paraphrases/) for our experiments. Follow their documentation to download the DIPPER paraphraser to /SCRATCH/DIR/ folder. Use recursive_paraphrase.py for for recursive paraphrasing.

> Input text: Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday.\nDetectives said three firearms, ammunition and a five-figure sum of money were recovered.\nA 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday...

> Watermarked text (detected): ...\nDetective Inspector Tom Wilson said: "Several firearms were recovered and a significant quantity of cash has been seized.\n"I would appeal to any witnesses or anyone who has any information about this incident to come forward so that we can bring those responsible to justice.\n"An extensive investigation is underway and we’re appealing for anyone with information to come forward. If you do and you think you may have any relevant information that could help with this investigation, or if you were in the area of Waterfront Park and Colonsay View at the time then please get in touch by calling 101 and quoting incident number 2811 of October 3.

> Paraphrased text (not detected): ...several firearms have been recovered and a significant amount of cash has been seized said inspector tom wilson. i'd like to appeal to any witnesses or to anyone who has information about this incident to come forward to bring them to justice. an extensive investigation is currently underway and we are appealing to any person with information to come forward. if you know any relevant information that could help with this investigation or you were in the waterfront park area and colonsay view area at the time then please contact the cia via the number 2811 of october 3
