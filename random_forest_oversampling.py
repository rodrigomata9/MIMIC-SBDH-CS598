from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline  # Use imblearn's pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np
import re

# === Load and Merge ===
labels_df = pd.read_csv("MIMIC-SBDH.csv")
text_df = pd.read_csv("social_history.csv")
df = labels_df.merge(text_df, left_on='row_id', right_on='ROW_ID', how='inner')

# === Drop rows with missing TEXT and reset index ===
df = df.dropna(subset=['TEXT']).reset_index(drop=True)

# === Clean Text ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

df['CLEAN_TEXT'] = df['TEXT'].apply(clean_text)

# === TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_all = vectorizer.fit_transform(df['CLEAN_TEXT'])

# === Classification Targets ===
binary_targets = {
    "Community-Present": df["sdoh_community_present"].astype(int),
    "Community-Absent": df["sdoh_community_absent"].astype(int),
    "Education": df["sdoh_education"].astype(int),
    "Economics": df["sdoh_economics"].astype(int),
    "Environment": df["sdoh_environment"].astype(int),
    "Alcohol Use": df["behavior_alcohol"].astype(int),
    "Tobacco Use": df["behavior_tobacco"].astype(int),
    "Drug Use": df["behavior_drug"].astype(int),
}

# === Evaluation Setup ===
results = []
scorer = make_scorer(f1_score, average='macro')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Main Loop with Oversampling ===
for label_name, y_series in binary_targets.items():
    y_raw = y_series.dropna()
    valid_indices = y_raw.index

    if y_raw.nunique() < 2:
        print(f"Skipping {label_name} due to insufficient class variety.")
        continue

    X = X_all[valid_indices]
    y = y_raw

    clf = Pipeline([
        ('oversample', RandomOverSampler(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_split=2, random_state=42))
    ])

    scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer)

    mean_score = scores.mean()
    std_score = scores.std()
    results.append([label_name, mean_score, std_score])

# === Print Results Table ===
print(f"{'SBDH':<20} {'Macro-F1':<12} {'Std Dev':<10}")
print("-" * 45)
for label_name, mean_score, std_score in results:
    print(f"{label_name:<20} {mean_score:.4f}      ± {std_score:.4f}")
