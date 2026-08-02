# English-Kashmiri Machine Translation (KATHE 2026)

A machine translation system for English-to-Kashmiri translation, built for the **KATHE 2026 competition** at NIT Srinagar.

---

## 🎯 Project Overview

**Competition:** KATHE 2026 (Kashmiri Language Technology Challenge)  
**Task:** English-to-Kashmiri Neural Machine Translation  
**Dataset:** BPCC (Bilingual Parallel Corpus Collection) - 98,929 English-Kashmiri pairs  
**Evaluation:** Geometric mean of BLEU and chrF++  
**Prize:** 30,000+ INR for top teams  
**In-person round:** Top 15–20 teams invited to NIT Srinagar (Aug 21)

---

## 📊 Competition Timeline

| Date | Milestone |
|------|-----------|
| Aug 1–17 | Main competition (16 days) |
| Aug 17, 11:59 PM | Final submission deadline |
| Aug 21 | In-person round at NIT Srinagar |

---

## 👥 Team

| Name | Role | Background |
|------|------|-----------|
| **Babar** | ML Architect | Model training & fine-tuning (Andrew Ng Course 1 complete) |
| **Muhaimin** | Data Engineer | Data cleaning & preprocessing (Courses 1 & 2 complete) |
| **Ilha** | DevOps | GitHub repo, evaluation scripts, experiment tracking |
| **Atif** | Kaggle Lead | Kaggle registration, submission management, CSV validation |

---

## 📁 Repository Structure

```
english_kashmiri-mt/
├── kathe_env/                  # Python virtual environment
├── notebooks/
│   ├── 01_baseline.py          # NLLB baseline (deprecated)
│   ├── 02_finetune.py          # IndicTrans2 fine-tuning (Kaggle)
│   ├── 03_experiments.py       # Hyperparameter experiments
│   └── 04_inference.py         # Final predictions on test set
├── data/
│   ├── raw/                    # Original BPCC dataset
│   ├── processed/              # Train/val/test splits (from Muhaimin)
│   └── predictions/            # Model outputs
├── models/
│   ├── baseline/               # NLLB checkpoints
│   └── finetuned/              # IndicTrans2 fine-tuned checkpoints
├── results/
│   ├── scores.json             # BLEU/chrF++ scores per experiment
│   └── examples.txt            # Sample predictions
├── .gitignore
├── LICENSE
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- GPU (recommended for training; using Kaggle Notebooks)
- HuggingFace account with `hf auth login`

### Setup

```bash
# Clone repo
git clone https://github.com/BabarZargar/english_kashmiri-mt.git
cd english_kashmiri-mt

# Create virtual environment
python3 -m venv kathe_env
source kathe_env/bin/activate  # On Windows: kathe_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace
hf auth login
```

### Load Dataset

```python
from datasets import load_dataset

# Load BPCC Kashmiri data (98,929 pairs)
dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest", split="kas_Arab")

print(f"Dataset size: {len(dataset)}")
print(f"English: {dataset[0]['src']}")
print(f"Kashmiri: {dataset[0]['tgt']}")
```

---

## 📈 Baseline Results

### NLLB-200 Baseline (Aug 2, 2026)

| Metric | Score |
|--------|-------|
| **BLEU** | 4.74 |
| **chrF++** | 34.48 |
| **Geometric mean** | 12.78 |
| **Model** | facebook/nllb-200-distilled-600M |
| **Test set** | First 100 BPCC examples |

**Status:** ❌ **Not viable**

**Reason:** NLLB-200 does not include Kashmiri in its 200-language portfolio. While it supports nearby languages (Hindi, Urdu, Punjabi), Kashmiri is absent. Output is gibberish Arabic script with no semantic meaning.

**Sample output:**
```
English:    "He bickers with the maids, harrows his hapless helper..."
Reference:  "سراپ دم ڈجے تڈیہیم شیڑنڈ یں ارک رے ل ذال..."
Prediction: "سراپ ریپ سرتش سن نریکوپں شیڑنڈ ڈریک وپ سریم ریل..."  ← Gibberish
```

**Decision:** Abandon NLLB. Proceed with **IndicTrans2**, which IS trained on Indian languages including Kashmiri.

---

## 🔧 Models Used

### Model 1: NLLB-200 (Baseline - DEPRECATED)

```python
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
```

**Why it failed:** No Kashmiri support.

---

### Model 2: IndicTrans2 (Active)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "ai4bharat/IndicTrans2-indic-indic"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="kas_Arab")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

**Why it works:**
- ✅ Trained on 10+ Indian languages (including Kashmiri)
- ✅ Handles morphologically rich languages well
- ✅ 1.3B parameters (fits on Kaggle GPU)
- ✅ Active Hugging Face support

---

## 📊 Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)
- **What:** Word n-gram overlap between prediction and reference
- **Why:** Standard MT metric for fluency
- **Target:** 30+
- **Formula:** Geometric mean of 1-gram to 4-gram precision

### chrF++ (Character n-gram F-score)
- **What:** Character-level F1 score
- **Why:** Critical for morphologically rich languages like Kashmiri (catches suffixes/prefixes)
- **Target:** 55+
- **Formula:** Harmonic mean of character n-gram precision and recall

### Geometric Mean
- **What:** √(BLEU × chrF++)
- **Why:** Balances both metrics equally
- **Target:** 40+
- **Formula:** `sqrt(bleu_score * chrf_score)`

**Why both metrics?** Kashmiri inflects heavily. A translation might have perfect word order (high BLEU) but wrong suffixes (low chrF++), or vice versa.

---

## 🛠️ Training Pipeline

### Week 1: Foundation (Aug 1–7) ✅
- [x] Learn NMT, BLEU/chrF++, seq2seq architecture
- [x] Set up local environment (Python, PyTorch, transformers)
- [x] Load BPCC dataset from HuggingFace
- [x] Run NLLB baseline (identify it's not suitable)

### Week 2: Fine-tuning (Aug 8–14) ⏳
- [ ] Get cleaned train/val/test data from Muhaimin
- [ ] Fine-tune IndicTrans2 on Kaggle Notebooks (GPU)
  - [ ] 2–3 epochs on training data
  - [ ] Validation set evaluation after each epoch
  - [ ] Save best checkpoint by geo_mean score
- [ ] Hyperparameter experiments
  - [ ] Learning rate: [1e-5, 5e-5, 1e-4]
  - [ ] Batch size: [16, 32, 64]
  - [ ] Epochs: [1, 2, 3]
- [ ] Log all results in `results/scores.json`

### Week 3: Submission (Aug 15–17) 📤
- [ ] Generate predictions on full test set
- [ ] Format predictions as CSV (English, Kashmiri)
- [ ] Make 3 Kaggle submissions (iterate if needed)
- [ ] Push final code to GitHub (public)
- [ ] Submit in-person round registration (if top 20)

---

## 🔬 Experiment Tracking

### Experiment Log Template

**File:** `results/scores.json`

```json
{
  "experiments": [
    {
      "id": "exp_001",
      "date": "2026-08-08",
      "model": "IndicTrans2-indic-indic",
      "config": {
        "learning_rate": 5e-5,
        "batch_size": 32,
        "epochs": 2,
        "warmup_steps": 500,
        "max_grad_norm": 1.0
      },
      "results": {
        "bleu": 22.5,
        "chrf": 45.3,
        "geo_mean": 31.9
      },
      "notes": "Good improvement over baseline. Try higher LR next."
    }
  ]
}
```

---

## 📝 Running Notebooks

### Local Development

```bash
# Baseline (NLLB) - DO NOT RUN UNLESS DEBUGGING
python notebooks/01_baseline.py

# Fine-tuning - USE KAGGLE NOTEBOOKS INSTEAD
# (GPUs not available on MacBook Air M4)
```

### Kaggle Notebooks (Recommended)

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Create new notebook
3. Add BPCC dataset as input
4. Copy `notebooks/02_finetune.py` code
5. Run with GPU (P100 or T4 available free)

**Why Kaggle?**
- ✅ GPU available (training ~100x faster)
- ✅ HuggingFace dependencies pre-installed
- ✅ No ONNX/custom code issues
- ✅ Easy submission to leaderboard

---

## 📦 Dependencies

### Core ML
- `torch>=2.0.0` — PyTorch for deep learning
- `transformers>=4.36.0` — HuggingFace models
- `datasets>=2.14.0` — HuggingFace datasets library
- `tokenizers>=0.14.0` — Fast tokenization

### Evaluation
- `sacrebleu>=2.4.0` — BLEU & chrF++ scoring
- `rouge-score>=0.1.2` — ROUGE metrics (optional)

### Development
- `jupyter>=1.0.0` — Jupyter notebooks
- `numpy>=1.24.0` — Numerical computing
- `pandas>=2.0.0` — Data manipulation

**Full list:** See `requirements.txt`

```bash
# Install all
pip install -r requirements.txt
```

---

## 🎓 Learning Resources

### Background Reading
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Encoder-decoder architecture
- [HuggingFace Course Chapter 1](https://huggingface.co/course/chapter1) — Transformers intro
- [Neural Machine Translation by Attention](https://arxiv.org/abs/1409.0473) — Seq2seq + attention (Bahdanau et al.)

### Fine-tuning References
- [HuggingFace Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [IndicTrans2 Paper](https://arxiv.org/abs/2305.16311) — AI4Bharat Indic translation model
- [BLEU Score Explained](https://en.wikipedia.org/wiki/BLEU) — Evaluation metric background

### Kashmiri-Specific
- [BPCC Dataset Paper](https://github.com/ai4bharat/BPCC) — Dataset documentation
- [Kashmiri on Wikipedia](https://en.wikipedia.org/wiki/Kashmiri_language) — Language facts

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: transformers.onnx`
**Solution:** This happens with IndicTrans2 on local machines. Use Kaggle Notebooks instead (dependencies pre-configured).

### Issue: NLLB outputs gibberish
**Solution:** Expected—NLLB doesn't support Kashmiri. Use IndicTrans2.

### Issue: `HuggingFace authentication failed`
**Solution:** Run `hf auth login` and paste your access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Issue: Out of memory (OOM) during training
**Solution:** Reduce batch size from 32 to 16. Or use Kaggle P100 GPU (more VRAM).

---

## 📝 Notes for In-Person Round

If your team makes top 15–20, you'll present at NIT Srinagar on Aug 21. Prepare:

1. **Approach:** Why IndicTrans2? How did you fine-tune?
2. **Results:** Final BLEU/chrF++ scores, comparison with baseline
3. **Challenges:** What went wrong? How did you debug?
4. **Future work:** How would you improve to 50+ BLEU?

---

## 📄 License

MIT License — See `LICENSE` file

---

## 🤝 Contributing

Team members:
- Push to `main` branch (not production-critical)
- Keep `notebooks/` clean (only working versions)
- Log all experiments in `results/scores.json`
- Update this README if you change structure/process

---

## 📞 Contact

- **GitHub:** [github.com/BabarZargar/english_kashmiri-mt](https://github.com/BabarZargar/english_kashmiri-mt)
- **Kaggle:** [kaggle.com/babarzargar](https://kaggle.com/babarzargar)

---

## 📌 Key Dates

| Date | Task | Owner |
|------|------|-------|
| Aug 2 | Baseline complete | Babar |
| Aug 8 | IndicTrans2 fine-tuning starts | Babar |
| Aug 14 | Best model selected | Babar |
| Aug 17 | 3 final submissions + code pushed | Atif & Babar |
| Aug 21 | In-person round (if qualified) | Team |

---

**Last updated:** Aug 2, 2026  
**Status:** 🟡 Week 1 complete, Week 2 starting

---

## Quick Reference

### Load Data
```python
from datasets import load_dataset
dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest", split="kas_Arab")
```

### Score Translations
```python
from sacrebleu import BLEU, CHRF
bleu = BLEU().corpus_score(predictions, [references])
chrf = CHRF().corpus_score(predictions, [references])
geo_mean = (bleu.score * chrf.score) ** 0.5
```

### Fine-tune Model
```python
from transformers import Trainer, TrainingArguments
trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
trainer.train()
```

---

**Good luck! 🚀 Let's beat that 40-point geometric mean!**
