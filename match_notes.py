import pandas as pd

# Path to your files
noteevents_path = "/mnt/c/users/rodri/Downloads/NOTEEVENTS.csv"
sbdh_path = "MIMIC-SBDH.csv"
output_path = "matched_discharge_notes.csv"

# Load the row_ids you want to extract (7025 total)
sbdh_df = pd.read_csv(sbdh_path)
target_ids = set(sbdh_df['row_id'].unique())

# Prepare a list to collect the matching notes
matched_notes = []

# Read NOTEEVENTS.csv in chunks
chunk_size = 100_000  # adjust based on memory capacity
for chunk in pd.read_csv(noteevents_path, chunksize=chunk_size):
    # Ensure column casing matches — change 'ROW_ID' if needed
    filtered = chunk[chunk['ROW_ID'].isin(target_ids)]
    if not filtered.empty:
        matched_notes.append(filtered[['ROW_ID', 'TEXT']])  # keep only what's needed

# Concatenate all matched chunks
result_df = pd.concat(matched_notes)

# Save to CSV
result_df.to_csv(output_path, index=False)

print(f"Saved {len(result_df)} matched notes to {output_path}")
