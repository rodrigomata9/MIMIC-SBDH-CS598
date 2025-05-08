import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from behavioral_testing import test_bank

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# === Define Test Cases ===
test_cases = test_bank

# === Evaluation Function ===
def evaluate_behavioral(model_path, test_samples):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    failures = 0
    for text, expected_label in test_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_label = torch.argmax(logits, dim=1).item()
        if pred_label != expected_label:
            failures += 1

    failure_rate = 100 * failures / len(test_samples)
    return failure_rate

# === Run Evaluation and Print Table with Example Test Case ===
print(f"{'SBDH':<18} {'Test':<15} {'Failure Rate (%)':<17} {'Example Test Case'}")
print("-" * 90)

for sbdh, tests in test_cases.items():
    model_name = sbdh.replace(" ", "_")
    model_path = f"./drive/MyDrive/saved_models/{model_name}_fold_0"

    if not os.path.exists(model_path):
        for test_type, samples in tests.items():
            example = samples[0][0]
            print(f"{sbdh:<18} {test_type:<15} {'N/A':<17} {example}")
        continue

    for test_type, samples in tests.items():
        example = samples[0][0]
        failure_rate = evaluate_behavioral(model_path, samples)
        print(f"{sbdh:<18} {test_type:<15} {failure_rate:>6.1f}%{'':10} {example}")
