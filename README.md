# From Trust to Lock-In: Human-Centered AI for Understanding Investment Fraud Narratives in Indian Digital Communities

[![ACM](https://img.shields.io/badge/ACM-SPAIC%20%4026-blue)](https://dl.acm.org/doi/10.1145/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📄 Paper

> **Gargi Sarkar, Arush Sachdeva, and Sandeep Kumar Shukla.** 2026. *From Trust to Lock-In: Human-Centered AI for Understanding Investment Fraud Narratives in Indian Digital Communities.* In *Proceedings of the 1st Security and Privacy for Asian Internet Communities Workshop (SPAIC), co-located with AsiaCCS 2026 (SPAIC '26)*. ACM, New York, NY, USA, 11 pages.

---

## 🔍 Overview

This repository contains the full experimental pipeline accompanying the above paper. The work addresses **investment fraud detection** in Indian digital communities, where scammers exploit trust, digital payment infrastructure, and social media platforms to extract money from victims.

The core contribution is a **multi-stage fraud journey classification system** — moving from simple keyword baselines through ensemble gradient boosting to transformer-based deep learning — on a real-world corpus of cybercrime complaints written in Romanized Telugu and code-mixed Indian English.

The five scam journey stages modeled are:

| Stage | Description |
|---|---|
| **Recruitment** | Initial contact via job offers, WhatsApp/Telegram groups, app invitations |
| **Trust** | Relationship building, mentorship framing, social engineering |
| **TriggerPayment** | First payment demand — deposits, trading fees, crypto/UPI transfers |
| **LockIn** | Victim trapped; withdrawals blocked, account frozen, pressure to deposit more |
| **ContinuedExtraction** | Repeated extortion, threats, demands to borrow from family/friends |

---

## 🗂 Repository Structure

```
.
├── Code.ipynb       # Main experimental notebook (Google Colab)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── Figures
```

---

## 📓 Notebook Walkthrough

The notebook is organized into the following major sections:

### 1. Data Loading & Preprocessing
- Loads `fraud_data_FINAL_FIXED.xlsx` containing English translations of cybercrime complaint narratives
- Handles missing values, lowercasing, special character removal, and whitespace tokenization
- Input column: `English_Translation`

### 2. Keyword-Based Label Engineering (Baseline)
Domain-informed keyword dictionaries are used to generate binary multi-label targets for each of the five fraud stages. Keywords cover fraud-specific vocabulary such as platform names (WhatsApp, Telegram, UPI, USDT, NEFT), financial terms, and psychological manipulation language.

### 3. TF-IDF + Logistic Regression Baseline
- `TfidfVectorizer` with unigrams/bigrams
- `MultiOutputClassifier` wrapping `LogisticRegression`
- 80/20 stratified train-test split
- Evaluation: per-label F1-score (baseline achieves ~0.32 F1 on the minority `ContinuedExtraction` stage)

### 4. Advanced Feature Engineering
A custom `AdvancedFeatureExtractor` class combines:
- TF-IDF word-level features
- TF-IDF character n-gram features
- Count-based (bag-of-words) features
- SBERT sentence embeddings (`sentence-transformers`)

### 5. Gradient Boosting Ensemble (Main ML Models)

Three models are trained and evaluated, all addressing class imbalance:

#### Model 1 — XGBoost with SMOTE
- `ImbalancedXGBoostClassifier`: per-label SMOTE oversampling + XGBoost
- Handles each output label independently

#### Model 2 — LightGBM with Focal Loss
- Custom focal loss (`alpha=0.25, gamma=2.0`) to down-weight easy negatives
- Per-label `LGBMClassifier` with `is_unbalance=True`

#### Model 3 — CatBoost with Threshold Optimization
- `ThresholdOptimizedCatBoost`: finds the decision threshold per label that maximizes F1 on the validation set
- Per-label threshold stored and applied at inference

#### Weighted Voting Ensemble
Combines all three models with weights `[0.35, 0.35, 0.30]` (XGBoost, LightGBM, CatBoost).

### 6. SHAP Explainability
- `TreeExplainer` on the XGBoost models for fast, stable SHAP values
- Per-stage top feature analysis
- Positive vs. negative SHAP feature directionality
- Summary statistics exported to `shap_summary_statistics.csv`

### 7. Transformer Models

#### DeBERTa (`microsoft/deberta-v3-base`)
- `AutoModelForSequenceClassification` configured for 5-label multi-label output
- Custom `FocalLoss` training criterion
- `AdamW` optimizer + linear scheduler with warmup
- Trained on GPU (CUDA) with `DataLoader` batching

#### RoBERTa (`roberta-base`)
- Same architecture and training configuration as DeBERTa
- Independent fine-tuning for ensemble diversity

### 8. Full Ensemble (Gradient Boosting + Transformers)
DeBERTa and RoBERTa probability outputs are integrated into the weighted voting ensemble alongside the three gradient boosting models for the final evaluation.

### 9. Visualizations (Camera-Ready ACM/IEEE Format)
- F1-score comparison heatmap across all models and stages
- Per-stage confusion matrices
- Radar chart of model performance profiles
- All figures formatted for ACM two-column layout

---

## ⚙️ Setup & Installation

### Option A — Google Colab (Recommended)
Open the notebook directly in Colab. All dependencies are installed in the first cell:

```bash
!pip install -q transformers datasets torch accelerate
!pip install -q sentence-transformers
!pip install -q imbalanced-learn
!pip install -q lightgbm catboost xgboost
!pip install -q shap
!pip install -q iterative-stratification
```

### Option B — Local Environment

```bash
git clone https://github.com/<your-org>/trust-to-lockin-fraud-detection.git
cd trust-to-lockin-fraud-detection
pip install -r requirements.txt
jupyter notebook From_Trust_to_Lock_In.ipynb
```

#### `requirements.txt`

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
lightgbm>=3.3
catboost>=1.2
shap>=0.42
imbalanced-learn>=0.10
sentence-transformers>=2.2
transformers>=4.30
torch>=2.0
datasets>=2.12
accelerate>=0.20
iterative-stratification>=0.1.7
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
```

---

## 📊 Data

The dataset that was analysed, contains cybercrime complaint narratives sourced from Indian law enforcement records. Complaint texts were originally written in Romanized Telugu and code-mixed Indian English, and were translated to English using a LLM-assisted pipeline (described in the companiont paper).

**Key fields:**

| Column | Description |
|---|---|
| `English_Translation` | Cleaned English translation of the complaint narrative |
| `Recruitment` | Binary label — 1 if complaint describes recruitment stage |
| `Trust` | Binary label — 1 if complaint describes trust-building stage |
| `TriggerPayment` | Binary label — 1 if complaint describes initial payment demand |
| `LockIn` | Binary label — 1 if complaint describes account lock-in / withdrawal block |
| `ContinuedExtraction` | Binary label — 1 if complaint describes ongoing extortion |

> **Note:** Due to privacy and legal sensitivity, the dataset may not be publicly distributable. Please contact the authors regarding data access.

---

## 📈 Key Results

| Model | Recruitment | Trust | TriggerPayment | LockIn | ContinuedExtraction | Macro F1 |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | — | — | — | — | ~0.32 | — |
| XGBoost + SMOTE | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper |
| LightGBM + Focal Loss | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper |
| CatBoost + Threshold Opt. | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper |
| **Ensemble (GB × 3)** | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper | **reported in paper** |
| **Full Ensemble (+ Transformers)** | reported in paper | reported in paper | reported in paper | reported in paper | reported in paper | **reported in paper** |

See the paper for full quantitative results and ablation analysis.

---

## 🧠 Method Summary

```
Raw Complaint Text (Romanized Telugu / Code-Mixed)
        │
        ▼
  LLM Translation Pipeline  ──►  English_Translation
        │
        ▼
  Text Preprocessing
  (lowercase, strip special chars, tokenize)
        │
        ▼
  Keyword-Based Label Generation  ──►  Binary Stage Labels (5)
        │
        ├──► TF-IDF + Logistic Regression (baseline)
        │
        ├──► Advanced Features (TF-IDF + SBERT)
        │         ├── XGBoost + SMOTE
        │         ├── LightGBM + Focal Loss
        │         └── CatBoost + Threshold Optimization
        │                     │
        │              Weighted Ensemble
        │
        └──► Transformer Fine-Tuning
                  ├── DeBERTa-v3-base
                  └── RoBERTa-base
                            │
                    Full Ensemble (all 5 models)
                            │
                            ▼
               SHAP Explainability + Visualizations
```

---

## 🔑 Key Design Decisions

**Why multi-label?** A single complaint can describe multiple stages of the fraud journey simultaneously. Binary or single-label classification would lose this sequential richness.

**Why keyword-based label generation?** Ground-truth human annotations at scale were unavailable. The keyword dictionaries were constructed from domain knowledge of Indian investment fraud patterns (Pig Butchering, part-time job scams, task-based fraud).

**Why focal loss?** The `ContinuedExtraction` stage is heavily underrepresented (~201 samples vs. majority classes). Standard cross-entropy loss leads the model to ignore minority labels. Focal loss and SMOTE both address this.

**Why SHAP?** SHAP values provide model-agnostic, locally faithful explanations that are required for human-centered analysis — understanding *why* a complaint is classified as a particular fraud stage is as important as the classification itself.

---

## 📜 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{sarkar2026trustlockin,
  author    = {Sarkar, Sargi and Sachdeva, Arush and Shukla, Sandeep Kumar},
  title     = {From Trust to Lock-In: Human-Centered AI for Understanding
               Investment Fraud Narratives in Indian Digital Communities},
  booktitle = {Proceedings of the 1st Security and Privacy for Asian Internet
               Communities Workshop (SPAIC), co-located with AsiaCCS 2026},
  series    = {SPAIC '26},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  pages     = {11},
  url       = {https://doi.org/10.1145/XXXXXXX.XXXXXXX}
}
```

---

## 📬 Contact

For questions about the paper, dataset access, or code, please open a GitHub issue or contact the authors through their institutional pages.

---

## ⚠️ Ethics & Responsible Use

This repository contains models and analysis built on real cybercrime complaint data. The work is intended strictly for **fraud detection research and victim protection**. Users must not repurpose any part of this work to facilitate, evade detection of, or study investment fraud from an adversarial standpoint. All data was handled in compliance with applicable privacy regulations.

---

## 📄 License

This code is released under the [MIT License](LICENSE). The dataset, if made available, is subject to separate terms — see the data access request process described above.
