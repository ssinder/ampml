import json
import os
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "APD6_antibacterial"
DATA_DIR_AF = PROJECT_ROOT / "data" / "APD6_antifungal"
OUTPUT_DIR = PROJECT_ROOT / "feature_extraction"

STANDARD_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def load_json_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data

def validate_sequence(sequence):
    if not sequence:
        return False, "Empty sequence"
    
    sequence_upper = sequence.upper()
    invalid_chars = [aa for aa in sequence_upper if aa not in STANDARD_AMINO_ACIDS]
    
    if invalid_chars:
        return False, f"Invalid amino acids: {set(invalid_chars)}"
    
    return True, "Valid"

def extract_sequences(json_file, output_fasta, label):
    data = load_json_data(json_file)
    
    records = []
    valid_count = 0
    invalid_count = 0
    invalid_sequences = []
    
    for entry in data:
        apd_id = entry.get('APD ID', 'Unknown')
        sequence = entry.get('Sequence', '').strip()
        
        is_valid, message = validate_sequence(sequence)
        
        if is_valid:
            record = SeqRecord(
                Seq(sequence),
                id=apd_id,
                description=f"Label:{label}"
            )
            records.append(record)
            valid_count += 1
        else:
            invalid_count += 1
            invalid_sequences.append({
                'APD_ID': apd_id,
                'Sequence': sequence,
                'Reason': message
            })
    
    with open(output_fasta, 'w') as fasta_file:
        SeqIO.write(records, fasta_file, 'fasta')
    
    print(f"Processed {json_file}:")
    print(f"  Valid sequences: {valid_count}")
    print(f"  Invalid sequences: {invalid_count}")
    print(f"  Output: {output_fasta}")
    
    if invalid_sequences:
        print(f"\nInvalid sequences:")
        for seq_info in invalid_sequences[:5]:
            print(f"  {seq_info['APD_ID']}: {seq_info['Reason']}")
        if len(invalid_sequences) > 5:
            print(f"  ... and {len(invalid_sequences) - 5} more")
    
    return valid_count, invalid_count

def create_test_dataset(antibacterial_json, antifungal_json, output_json, n_per_class=10):
    antibacterial_data = load_json_data(antibacterial_json)
    antifungal_data = load_json_data(antifungal_json)
    
    test_data = []
    
    for i, entry in enumerate(antibacterial_data[:n_per_class]):
        entry['Label'] = 0
        test_data.append(entry)
    
    for i, entry in enumerate(antifungal_data[:n_per_class]):
        entry['Label'] = 1
        test_data.append(entry)
    
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(test_data, json_file, indent=2, ensure_ascii=False)
    
    print(f"Created test dataset: {output_json}")
    print(f"  Antibacterial: {n_per_class}")
    print(f"  Antifungal: {n_per_class}")
    print(f"  Total: {len(test_data)}")
    
    return test_data

def main():
    antibacterial_json = DATA_DIR / 'antibacterial_peptides.json'
    antifungal_json = DATA_DIR_AF / 'antifungal_peptides.json'
    
    antibacterial_fasta = OUTPUT_DIR / 'antibacterial_sequences.fasta'
    antifungal_fasta = OUTPUT_DIR / 'antifungal_sequences.fasta'
    
    test_json = OUTPUT_DIR / 'test_sequences.json'
    test_fasta = OUTPUT_DIR / 'test_sequences.fasta'
    
    print("=" * 60)
    print("Data Preprocessing")
    print("=" * 60)
    
    print("\n1. Processing antibacterial peptides...")
    ab_valid, ab_invalid = extract_sequences(
        antibacterial_json, 
        antibacterial_fasta, 
        label=0
    )
    
    print("\n2. Processing antifungal peptides...")
    af_valid, af_invalid = extract_sequences(
        antifungal_json, 
        antifungal_fasta, 
        label=1
    )
    
    print("\n3. Creating test dataset...")
    test_data = create_test_dataset(
        antibacterial_json,
        antifungal_json,
        test_json,
        n_per_class=10
    )
    
    print("\n4. Creating test FASTA file...")
    records = []
    for entry in test_data:
        apd_id = entry.get('APD ID', 'Unknown')
        sequence = entry.get('Sequence', '').strip()
        label = entry.get('Label', 0)
        
        record = SeqRecord(
            Seq(sequence),
            id=apd_id,
            description=f"Label:{label}"
        )
        records.append(record)
    
    with open(test_fasta, 'w') as fasta_file:
        SeqIO.write(records, fasta_file, 'fasta')
    
    print(f"  Output: {test_fasta}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Antibacterial: {ab_valid} valid, {ab_invalid} invalid")
    print(f"Antifungal: {af_valid} valid, {af_invalid} invalid")
    print(f"Test dataset: {len(test_data)} sequences")
    print("=" * 60)

if __name__ == "__main__":
    main()
