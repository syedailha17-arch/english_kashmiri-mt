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
| **Babar** | ML Architect | Model training & fine-tuning |
| **Muhaimin** | Data Engineer | Data cleaning & preprocessing |
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
git clone https://github.com/syedailha17-arch/english_kashmiri-mt.git
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

*Team GitHub profiles:*
- Babar: [github.com/BabarZargar](https://github.com/BabarZargar)
- Muhaimin: [github.com/Muha1m1n](https://github.com/Muha1m1n)
- Ilha: [github.com/syedailha17-arch](https://github.com/syedailha17-arch)
- Aatif:[github.com/Aatif-wani](https://github.com/Aatif-wani)
