import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from behavioral.behavioral_testing import test_bank  # Assuming this contains your test cases

# === Load and prepare the original model ===
def load_and_prepare_model():
    # Load your original data (same as your RF training code)
    labels_df = pd.read_csv("MIMIC-SBDH.csv")
    text_df = pd.read_csv("social_history.csv")
    df = labels_df.merge(text_df, left_on='row_id', right_on='ROW_ID', how='inner')
    df = df.dropna(subset=['TEXT']).reset_index(drop=True)
    
    # Clean text function
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text
    
    df['CLEAN_TEXT'] = df['TEXT'].apply(clean_text)
    
    # Initialize vectorizer and fit on all text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(df['CLEAN_TEXT'])
    
    return vectorizer

# === Behavioral Testing Function ===
def run_behavioral_tests(vectorizer):
    # Initialize RF model with same params as original
    
    rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_split=2, random_state=42)
    
    # Train on a small subset (just to initialize weights)
    # In practice, you should use your actual trained model
    dummy_X = vectorizer.transform(["dummy text"])
    dummy_y = [0]
    rf_model.fit(dummy_X, dummy_y)
    
    results = []
    
    for sbdh, tests in test_bank.items():
        for test_type, examples in tests.items():
            correct = 0
            total = len(examples)
            
            for text, true_label in examples:
                # Preprocess the test text
                clean_text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
                X_test = vectorizer.transform([clean_text])
                
                # Get prediction
                pred = rf_model.predict(X_test)[0]
                
                if pred == true_label:
                    correct += 1
            print(correct)
            accuracy = correct *1.0 / total
            failure_rate = (1 - accuracy) * 100  # Convert to percentage
            
            results.append({
                'SBDH': sbdh,
                'Test': test_type,
                'Failure Rate': f"{failure_rate:.1f}%",
                'Examples Tested': total
            })
    
    return pd.DataFrame(results)

# === Main Execution ===
if __name__ == "__main__":
    print("Loading and preparing vectorizer...")
    vectorizer = load_and_prepare_model()
    
    print("\nRunning behavioral tests...")
    test_results = run_behavioral_tests(vectorizer)
    
    print("\n=== Behavioral Test Results ===")
    print(test_results.to_markdown(index=False))