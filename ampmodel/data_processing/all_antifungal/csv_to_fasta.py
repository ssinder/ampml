import pandas as pd
import os

data_dir = r'd:\trae\paper\data\all_antifungal'

print("Reading merged_antifungal_le60_final.csv...")
df = pd.read_csv(os.path.join(data_dir, 'merged_antifungal_le60_final.csv'), encoding='latin1')

print(f"Total rows: {len(df)}")

fasta_entries = []

for idx, row in df.iterrows():
    sequence = row['Sequence']
    seq_id = str(idx + 1)
    fasta_entries.append(f">{seq_id}\n{sequence}")

print(f"Generated {len(fasta_entries)} FASTA entries")

output_path = os.path.join(data_dir, 'merged_antifungal_le60.fasta')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(fasta_entries))

print(f"\nFASTA file saved to: {output_path}")

print("\n=== Sample entries ===")
for i in range(min(5, len(fasta_entries))):
    print(fasta_entries[i])
    print()

print("\n=== Summary ===")
print(f"Total sequences: {len(fasta_entries)}")
print(f"Output file: {output_path}")

total_length = sum(len(entry.split('\n')[1]) for entry in fasta_entries)
print(f"Total amino acids: {total_length}")
print(f"Average sequence length: {total_length/len(fasta_entries):.2f}")
