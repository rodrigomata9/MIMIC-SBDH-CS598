import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from behavioral.behavioral_testing import test_bank
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def load_and_train_model():
    # Load and prepare data
    labels_df = pd.read_csv("MIMIC-SBDH.csv")
    text_df = pd.read_csv("social_history.csv")
    df = labels_df.merge(text_df, left_on='row_id', right_on='ROW_ID', how='inner')
    df = df.dropna(subset=['TEXT']).reset_index(drop=True)
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text
    
    df['CLEAN_TEXT'] = df['TEXT'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_all = vectorizer.fit_transform(df['CLEAN_TEXT'])
    
    # Train one model per SBDH
    models = {}
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
    
    for label_name, y_series in binary_targets.items():
        y = y_series.dropna()
        X = X_all[y.index]
        
        if y.nunique() < 2:
            print(f"Skipping {label_name} due to insufficient class variety.")
            continue
            
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('xgb', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                reg_lambda=1,
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        pipeline.fit(X, y)
        models[label_name] = pipeline
    
    return vectorizer, models

def run_xgboost_behavioral_tests(vectorizer, models):
    results = []
    
    for sbdh, tests in test_bank.items():
        if sbdh not in models:
            continue
            
        for test_type, examples in tests.items():
            correct = 0
            total = len(examples)
            
            for text, true_label in examples:
                clean_text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
                X_test = vectorizer.transform([clean_text])
                pred = models[sbdh].predict(X_test)[0]
                if pred == true_label:
                    correct += 1
            
            failure_rate = (1 - (correct / total)) * 100
            results.append({
                'SBDH': sbdh,
                'Test': test_type,
                'Failure Rate': f"{failure_rate:.1f}%",
                'Examples Tested': total
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Loading data and training XGBoost models...")
    vectorizer, models = load_and_train_model()
    
    print("\nRunning behavioral tests...")
    xgboost_test_results = run_xgboost_behavioral_tests(vectorizer, models)
    
    print("\n=== XGBoost Behavioral Test Results ===")
    print(xgboost_test_results.to_markdown(index=False))