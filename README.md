
# MIMIC-SBDH Reproduction and Analysis

This repository contains a reproducibility study and ablation analysis of the paper **"MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health"**. We re-implement and evaluate models including Random Forest, XGBoost, and Bio-ClinicalBERT using discharge summaries from MIMIC-III. We also include behavioral testing, oversampling ablations, and a lightweight alternative using CountVectorizer.

---

## Project Goals

- Reproduce macro-F1 results from the original paper.
- Evaluate class imbalance with and without oversampling.
- Perform behavioral testing to reveal failure modes not captured by traditional metrics.
- Conduct an ablation using a simpler vectorizer for Random Forest.

---

## Models Implemented

- `random_forest.py`: TF-IDF + Random Forest
- `XGBoost_oversampling.py`: TF-IDF + XGBoost + SMOTE oversampling
- `bio-clinicalBERT.py`: Fine-tuned transformer on clinical notes
- `ablation.py`: CountVectorizer + Random Forest (lightweight baseline)
- `behavioral_testing.py`: Edge case testing using CheckList-style test banks

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/rodrigomata9/MIMIC-SBDH-CS598.git
cd mimic-sbdh-reproduction
```

### 2. Create a Virtual Environment

```bash
python -m venv env
source env/bin/activate 
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Setup

1. Download **NOTEEVENTS.csv** from MIMIC-III (requires access) or [kaggle](https://www.kaggle.com/datasets/hussameldinanwer/noteevents-mimic-iii).
2. Download **MIMIC-SBDH.csv** and **MIMIC-SBDH-keywords.csv** from the [original research repository](https://github.com/hibaahsan/MIMIC-SBDH).
2. Place `NOTEEVENTS.csv` in this directory alongside `MIMIC-SBDH.csv` and `MIMIC-SBDH-keywords.csv`.
3. Run `match_notes.py` to extract discharge notes. This should produce `matched_discharge_notes.csv`.
3. Run `clean_matched_notes.py` to extract just the Social History section. This should produce `social_history.csv`, the cleaned discharge summaries for model input

---

## Running Models

### Random Forest

```bash
python random_forest.py
python random_forest_oversampling.py
```

Results will be printed directly the stdout.

### XGBoost

```bash
python XGBoost_oversampling.py
```
Results will be printed directly the stdout.

### Bio-ClinicalBERT

```bash
python bio-clinicalBERT.py
python bio-clinicalBERT_oversampling.py
```
Results will be printed directly the stdout.

### Lightweight Ablation (CountVectorizer)

```bash
python ablation.py
```
Results will be printed directly the stdout.

---

## Behavioral Testing

Random Forest
```bash
python behavioral_test_rf.py
```

XGBoost
```bash
python xgboost_behavioral.py
```

BCB
```bash
python bcb_behavioral.py
```

Our test cases are saved in `behavioral_testing.py`. You can use it to import and validate a trained model on synthetic test cases:

- Negations
- Misspellings
- Historical context (for alcohol, tobacco, and drug)
- Attribution errors (for alcohol, tobacco, and drug)

---

## Evaluation

All models are evaluated using macro-F1 with `StratifiedKFold` CV. Results are printed to the console.

---

## Citation

If you use this code, please cite the original MIMIC-SBDH paper:

```
@inproceedings{mimic_sbdh_2021,
  title={MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health},
  author={Anwer, Hussameldin and others},
  booktitle={Proceedings of AAAI},
  year={2021}
}
```

---

## License

This project is released for academic use. Please refer to the license terms of the MIMIC-III database.

---

## Links

- [Video](https://mediaspace.illinois.edu/media/t/1_j9h794fg)
- [Original Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8734043/)
