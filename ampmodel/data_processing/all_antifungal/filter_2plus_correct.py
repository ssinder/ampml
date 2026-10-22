import pandas as pd
import os
from collections import defaultdict

data_dir = r'd:\trae\paper\data\all_antifungal'

print("Reading filtered CSV file...")
filtered_df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_filtered.csv'))

print(f"Total sequences in filtered CSV: {len(filtered_df)}")

files = {
    'Antifungal_amps.csv': 'Sequence',
    'antifungal.csv': 'Seq',
    'antifungal_peptides.csv': 'Sequence',
    'merged_data.csv': 'Seqence',
    'peptides_merged.csv': 'SEQUENCE'
}

standard_amino_acids = set('ARNDCEQGHILKMFPSTWYV')

def filter_standard_aa(sequence):
    for char in sequence:
        if char not in standard_amino_acids:
            return False
    return True

seq_file_sets = defaultdict(set)

for filename, seq_col in files.items():
    filepath = os.path.join(data_dir, filename)
    print(f"\nReading {filename}...")
    df = pd.read_csv(filepath)
    
    if seq_col in df.columns:
        for seq in df[seq_col].dropna():
            seq_upper = seq.upper()
            if filter_standard_aa(seq):
                seq_file_sets[seq_upper].add(filename)

print(f"\nTotal unique sequences found: {len(seq_file_sets)}")

sequences_in_2plus = [seq for seq, fileset in seq_file_sets.items() if len(fileset) >= 2]
print(f"Sequences appearing in 2+ files: {len(sequences_in_2plus)}")

sequences_2plus_set = set(sequences_in_2plus)

filtered_2plus_df = filtered_df[filtered_df['Sequence'].isin(sequences_2plus_set)].copy()

print(f"\nRows after filtering (2+ files): {len(filtered_2plus_df)}")

occurrence_count = defaultdict(int)
for seq, fileset in seq_file_sets.items():
    if len(fileset) >= 2:
        occurrence_count[len(fileset)] += 1

print("\n=== Distribution of sequences appearing in 2+ files ===")
for count in sorted(occurrence_count.keys()):
    print(f"Sequences appearing in {count} file(s): {occurrence_count[count]}")

output_path = os.path.join(data_dir, 'merged_antifungal_2plus_correct.csv')
filtered_2plus_df.to_csv(output_path, index=False)
print(f"\nCSV saved to: {output_path}")

print("\n=== Summary ===")
print(f"Original filtered sequences: {len(filtered_df)}")
print(f"Unique sequences in 2+ files: {len(sequences_in_2plus)}")
print(f"Rows in output CSV: {len(filtered_2plus_df)}")
print(f"Removed sequences (appearing in only 1 file): {len(filtered_df) - len(filtered_2plus_df)}")
