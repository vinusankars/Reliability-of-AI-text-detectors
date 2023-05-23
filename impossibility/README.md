# Impossibility Result

## AUROC bound

To plot the AUROC bound, run:
```
python roc_bound.py
```

## Total Variation Estimation

**Step 1:** Download GPT-2 and GPT-3 output datasets into `data` directory.

**Step 2:** Train RoBERTa estimator using:
```
python roberta_tv.py [human-dataset name] [ai-dataset name] --data_dir [path to datasets] --tv_model roberta-large --batch_size [num] --seq_len [num]
```
For example, to estimate TV between `webtext_completion` and `text-ada-001_completion` datasets, use:
```
python roberta_tv.py webtext_completion text-ada-001_completion --data_dir data/webtext-completion-dataset --tv_model roberta-large --batch_size 20 --seq_len 50
```
This step will save trained TV estimation models in the `models` directory.

**Step 3:** Evaluate TV using the trained model:
```
python eval_tv.py [human-dataset name] --model_name roberta-large --data_dir [path to datasets]
```
Set the `GPT_DS` list in `eval_tv.py` appropriately based on the AI text datasets to be used for the evaluations.
This step will save TV estimates in a `.json` file under the `results` directory.

**Step 4:** Plot the TV estimates:
```
python plot_tv.py [path to above JSON file]
```
This will save the plot in a `.png` file in the same location as the `.json` file.