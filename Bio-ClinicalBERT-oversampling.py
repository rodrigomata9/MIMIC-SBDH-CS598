import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
import json
import os

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
    "Community-Present": df["sdoh_community_present"].astype(int),
    "Community-Absent": df["sdoh_community_absent"].astype(int),
    "Education": df["sdoh_education"].astype(int),
    "Economics": df["sdoh_economics"].astype(int),
    "Environment": df["sdoh_environment"].astype(int),
    "Alcohol Use": df["behavior_alcohol"].astype(int),
    "Drug Use": df["behavior_drug"].astype(int),  # Fixed the column name (was incorrectly using alcohol)
    "Tobacco Use": df["behavior_tobacco"].astype(int),
}

# === Function to perform oversampling ===
def apply_oversampling(X_text, y):
    """
    Apply oversampling to the text data based on labels.

    For text data, we need a different approach than traditional SMOTE:
    1. Create a simple numerical feature (length) to use with RandomOverSampler
    2. Apply oversampling to these features
    3. Use the resampled indices to get the corresponding text data
    """
    # Create a simple feature matrix (text length as a feature)
    X_features = pd.DataFrame({'text_length': X_text.apply(len)})

    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res_features, y_res = ros.fit_resample(X_features, y)

    # Get the original texts for the resampled indices (including duplicates for minority classes)
    X_res_text = []
    for idx in ros.sample_indices_:
        if idx < len(X_text):  # Original sample
            X_res_text.append(X_text.iloc[idx])
        else:  # This is a synthetic sample - use the original from which it was generated
            origin_idx = ros.sample_indices_[idx - len(X_text)]
            X_res_text.append(X_text.iloc[origin_idx])

    # Convert back to pandas Series with proper index
    X_res_text = pd.Series(X_res_text)

    print(f"Original class distribution: {pd.Series(y).value_counts()}")
    print(f"Resampled class distribution: {pd.Series(y_res).value_counts()}")

    return X_res_text, y_res

# === 2-Fold Cross-Validation per SBDH ===
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Create directory for saving models
os.makedirs("./drive/MyDrive/saved_models", exist_ok=True)

# Dictionary to store results
all_results = {}

for label_name, y_series in binary_targets.items():
    print(f"\n{'='*50}")
    print(f"Training Bio-ClinicalBERT on: {label_name}")
    print(f"{'='*50}")

    y_series = y_series.dropna()
    indices = y_series.index
    X_text = df.loc[indices, 'CLEAN_TEXT']
    y = y_series

    if y.nunique() < 2:
        print(f"Skipping {label_name} due to insufficient class variety.")
        continue

    # Initialize results storage
    results = {"f1_scores": [], "per_class_f1": {}}

    # Run with oversampling
    print("\n--- Training with oversampling ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
        print(f"Fold {fold + 1}/2")
        X_train, X_val = X_text.iloc[train_idx], X_text.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Check class distribution before oversampling
        print(f"Train class distribution before oversampling: {pd.Series(y_train).value_counts()}")

        # Apply oversampling to training data only
        X_train_resampled, y_train_resampled = apply_oversampling(X_train, y_train)

        train_dataset = SBDHDataset(X_train_resampled, y_train_resampled)
        val_dataset = SBDHDataset(X_val, y_val)

        # Get number of unique labels
        num_labels = len(np.unique(y))

        model = AutoModelForSequenceClassification.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            num_labels=num_labels,
            classifier_dropout=0.2
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{label_name.replace(' ', '_')}/fold_{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        # Save the model
        model_save_path = f"./drive/MyDrive/saved_models/{label_name.replace(' ', '_')}_fold_{fold}"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        # Evaluation
        preds = trainer.predict(val_dataset)
        pred_labels = np.argmax(preds.predictions, axis=1)
        macro_f1 = f1_score(y_val, pred_labels, average="macro")
        results["f1_scores"].append(macro_f1)

        # Per-class F1 scores
        class_report = classification_report(y_val, pred_labels, output_dict=True)
        for class_idx in np.unique(y_val):
            class_name = str(class_idx)
            if class_name not in results["per_class_f1"]:
                results["per_class_f1"][class_name] = []
            results["per_class_f1"][class_name].append(class_report[class_name]['f1-score'])

        # Save evaluation results
        fold_results = {
            "macro_f1_score": macro_f1,
            "per_class_f1": {str(class_idx): class_report[str(class_idx)]['f1-score'] for class_idx in np.unique(y_val)},
            "label_name": label_name,
            "fold": fold
        }

        os.makedirs(model_save_path, exist_ok=True)
        with open(f"{model_save_path}/eval_results.json", "w") as f:
            json.dump(fold_results, f)

    # Print and store summary of results for this SBDH
    print(f"\nSummary for {label_name}:")
    print(f"Macro-F1: {np.mean(results['f1_scores']):.4f} ± {np.std(results['f1_scores']):.4f}")

    # Per-class summaries
    print("\nPer-class F1 scores:")
    for class_name, scores in results["per_class_f1"].items():
        print(f"  Class {class_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Store results for this label
    all_results[label_name] = results

# Save overall results to JSON
with open("./drive/MyDrive/saved_models/oversampling_results.json", "w") as f:
    # Convert numpy values to Python native types for JSON serialization
    json_serializable_results = {}
    for label_name, label_results in all_results.items():
        json_serializable_results[label_name] = {
            "f1_scores": [float(score) for score in label_results["f1_scores"]],
            "per_class_f1": {
                class_name: [float(score) for score in scores]
                for class_name, scores in label_results["per_class_f1"].items()
            }
        }
    json.dump(json_serializable_results, f, indent=2)

print("\nExperiment completed. Results saved to oversampling_results.json")