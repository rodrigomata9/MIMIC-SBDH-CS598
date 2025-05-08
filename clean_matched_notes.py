import pandas as pd
import re

df = pd.read_csv('matched_discharge_notes.csv')

def extract_social_history(text):
    if pd.isna(text):
        return ""
    
    # Try to find social history using regex with case insensitivity
    pattern = r"(?:SOCIAL HISTORY:|Social History:)(.*?)(?=\n\s*(?:[A-Z][A-Z\s]+:|[A-Z][a-z]+\s+[A-Z][a-z]+:)|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # As a backup, try the section-splitting approach
    sections = re.split(r"\n\s*\n", text)
    for section in sections:
        if re.match(r"(?:SOCIAL HISTORY:|Social History:)", section.strip(), re.IGNORECASE):
            return re.sub(r"(?:SOCIAL HISTORY:|Social History:)", "", section.strip(), flags=re.IGNORECASE, count=1).strip()
    
    return ""

df['TEXT'] = df['TEXT'].apply(extract_social_history)

print(f"df is size {len(df)}")

df.to_csv('social_history.csv', index=False)

print(f"Extraction complete. File saved as 'social_history.csv' with {len(df)} non-empty entries.")