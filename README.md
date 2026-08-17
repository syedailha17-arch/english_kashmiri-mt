# English--Kashmiri Machine Translation (KATHE 2026)

An English-to-Kashmiri neural machine translation system developed for the **KATHE 2026: Kashmiri Language Technology Challenge** at NIT Srinagar.

## Results

| Experiment | Fine-tuning approach | Competition leaderboard score |
|---|---|---:|
| IndicTrans2 1B | LoRA (parameter-efficient) fine-tuning | 7.90 |
| IndicTrans2 200M | Full fine-tuning | **15.55** |

The final submission uses the **IndicTrans2 200M** model. It achieved the best score of **15.55**.

## Team

| Name | Role |
|---|---|
| Babar | Model training and fine-tuning |
| Muhaimin | Data cleaning and preprocessing |
| Ilha | Repository and experiment tracking |
| Atif | Kaggle submissions and CSV validation |

## Methodology

### Model selection

We first experimented with an IndicTrans2 1B model. Because of the memory requirements of this larger model, it was trained with **LoRA**, a parameter-efficient fine-tuning method that updates only a small set of trainable adapter parameters instead of the full model. This experiment obtained a leaderboard score of **7.90**.

We then switched to [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M). Its smaller size made it feasible to fine-tune the model's full set of parameters on the available Kaggle GPU. Full fine-tuning produced a substantially stronger result, with a leaderboard score of **15.55**, so this became the final model.

### Frameworks and libraries

The project uses **PyTorch**, **Hugging Face Transformers**, and **IndicTransToolkit**. Pandas and NumPy are used for data handling, and tqdm is used to display inference progress. Training and inference were performed in a GPU-enabled Kaggle Notebook.

### Data

The training corpus combines:

- the BPCC English--Kashmiri parallel corpus;
- Syed Matla-ul-Qamar's Hugging Face English--Kashmiri dataset, containing approximately 30,000 sentence pairs; and
- the competition-provided training data where applicable.

Rows with missing or empty English/Kashmiri text were removed. To control sequence length, BPCC pairs were restricted to at most 150 whitespace-separated tokens on each side, while the additional Hugging Face pairs were restricted to at most 50. After cleaning, the final training set contained **105,665** pairs: **75,913** BPCC pairs and **29,752** pairs from the additional Hugging Face dataset.

### Preprocessing

The source language was processed as `eng_Latn` and the target language as `kas_Arab` using IndicTransToolkit. Source and target text were tokenized with truncation at a maximum length of 128. Dynamic padding was applied by the sequence-to-sequence data collator, with padding to a multiple of 8 for efficient GPU computation.

### Full fine-tuning configuration

The final 200M model was fine-tuned using Hugging Face `Seq2SeqTrainer`.

| Setting | Value |
|---|---:|
| Epochs | 5 |
| Per-device training batch size | 16 |
| Gradient accumulation | 2 steps |
| Effective batch size | 32 |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Warm-up ratio | 0.1 |
| Learning-rate scheduler | Cosine |
| Precision | bfloat16 |
| Checkpoint frequency | Every epoch |
| Checkpoints retained | 5 |

Checkpoints are saved in `/kaggle/working/indictrans2-200m-fullft`. The final model and tokenizer are saved in `/kaggle/working/best-model-200m`.

### Inference and candidate submissions

For each English test sentence, the model applies IndicTransToolkit preprocessing, generates Kashmiri text using beam search, and postprocesses the result as `kas_Arab`. We generated candidate submissions from the final model with these decoding settings:

- 5 beams, length penalty 1.0;
- 10 beams, length penalty 0.8;
- 10 beams, length penalty 1.0; and
- 8 beams, length penalty 0.7.

We also generated outputs from every epoch checkpoint using 5 beams with a length penalty of 1.0. Each candidate is saved as a CSV with the required `ID` and `kashmiri_text` columns, so checkpoints and decoding settings can be compared on the competition leaderboard.

## Reproducibility

The combined training set is shuffled with `random_state=42`. Before ending a Kaggle session, the final model, epoch checkpoints, and submission CSV files must be stored under `/kaggle/working/`, then preserved by creating a Kaggle notebook version with **Save output files** enabled.

## Main dependencies

```text
torch
transformers
IndicTransToolkit
pandas
numpy
tqdm
```

## Notes

- The Kaggle Notebook requires GPU acceleration to fine-tune the model efficiently.
- A Hugging Face token is configured as the Kaggle secret `HF_TOKEN` when access is required to download the base model.
- The competition's official leaderboard determines the reported scores above.

## License

MIT License.
