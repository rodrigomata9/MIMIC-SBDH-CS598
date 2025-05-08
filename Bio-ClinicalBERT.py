import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset

# === Load and Merge ===
labels_df = pd.read_csv("MIMIC-SBDH.csv")
text_df = pd.read_csv("social_history.csv")
df = labels_df.merge(text_df, left_on='row_id', right_on='ROW_ID', how='inner')
df = df.dropna(subset=['TEXT']).reset_index(drop=True)

# === Clean Text ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

df['CLEAN_TEXT'] = df['TEXT'].apply(clean_text)

# === Setup Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# === Dataset Class ===
class SBDHDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=256)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# === Targets to Train ===
binary_targets = {
    # "Community-Present": df["sdoh_community_present"].astype(int),
    # "Community-Absent": df["sdoh_community_absent"].astype(int),
    #"Education": df["sdoh_education"].astype(int),
    #"Economics": df["sdoh_economics"].astype(int),
    #"Environment": df["sdoh_environment"].astype(int),
    "Alcohol Use": df["behavior_alcohol"].astype(int),
    "Drug Use": df["behavior_alcohol"].astype(int),
    "Tobacco Use": df["behavior_tobacco"].astype(int),
}

# === 2-Fold Cross-Validation per SBDH ===
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
import json
for label_name, y_series in binary_targets.items():
    print(f"\nTraining Bio-ClinicalBERT on: {label_name}")
    y_series = y_series.dropna()
    indices = y_series.index
    X_text = df.loc[indices, 'CLEAN_TEXT']
    y = y_series

    if y.nunique() < 2:
        print(f"Skipping {label_name} due to insufficient class variety.")
        continue

    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
        print(f"Fold {fold + 1}/1")
        X_train, X_val = X_text.iloc[train_idx], X_text.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_dataset = SBDHDataset(X_train, y_train)
        val_dataset = SBDHDataset(X_val, y_val)

        model = AutoModelForSequenceClassification.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            num_labels=3
        )
        print("TrainingArguments source:", TrainingArguments.__module__)
        # Add to your model initialization
        model = AutoModelForSequenceClassification.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            num_labels=5,
            classifier_dropout=0.2  # Add dropout
        )

        # Improved training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{label_name.replace(' ', '_')}/fold_{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,  # Reduced from 5
            learning_rate=2e-5,  # Lower learning rate
            weight_decay=0.01,   # Add weight decay for regularization
            eval_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",  # Save each epoch
            load_best_model_at_end=True,  # Load best model at end of training
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        # Save the model, tokenizer and configuration
        model_save_path = f"./drive/MyDrive/saved_models/{label_name.replace(' ', '_')}_fold_{fold}"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)


        # Evaluation
        preds = trainer.predict(val_dataset)
        pred_labels = np.argmax(preds.predictions, axis=1)
        f1 = f1_score(y_val, pred_labels, average="macro")
        f1_scores.append(f1)
        # Save evaluation results
        fold_results = {
            "f1_score": f1,
            "label_name": label_name,
            "fold": fold
        }
        with open(f"{model_save_path}/eval_results.json", "w") as f:
            json.dump(fold_results, f)

    print(f"{label_name} — Macro-F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
