import pandas as pd
import numpy as np
from collections import Counter

rep_csv_file = r'D:\trae\paper\data\uniprot_peptide\mmseq\representative_sequences.csv'
output_file = r'D:\trae\paper\data\uniprot_peptide\mmseq\representative_sequence_analysis.txt'

df = pd.read_csv(rep_csv_file)
sequences = df['sequence'].tolist()

print("=" * 70)
print("代表性序列特征分析")
print("=" * 70)

print(f"\n总序列数: {len(sequences)}")

lengths = [len(seq) for seq in sequences]
print(f"\n序列长度统计:")
print(f"  最小长度: {min(lengths)} aa")
print(f"  最大长度: {max(lengths)} aa")
print(f"  平均长度: {np.mean(lengths):.2f} aa")
print(f"  中位数长度: {np.median(lengths):.2f} aa")
print(f"  标准差: {np.std(lengths):.2f} aa")

length_ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100), (101, 150), (151, 200)]
range_labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", "101-150", "151-200"]

print(f"\n序列长度分布:")
for (min_len, max_len), label in zip(length_ranges, range_labels):
    count = sum(1 for length in lengths if min_len <= length <= max_len)
    percentage = count / len(lengths) * 100
    if count > 0:
        print(f"  {label:>8}: {count:>4} 条 ({percentage:>5.1f}%)")

all_amino_acids = ''.join(sequences)
aa_counts = Counter(all_amino_acids)
total_aa = len(all_amino_acids)

print(f"\n氨基酸组成分析:")
print(f"  总氨基酸数: {total_aa}")
print(f"  氨基酸种类: {len(aa_counts)}")

print("\n各氨基酸出现频率 (前10):")
for aa, count in aa_counts.most_common(10):
    percentage = count / total_aa * 100
    print(f"  {aa}: {count:>6} ({percentage:>5.2f}%)")

if 'X' in aa_counts:
    print(f"\n注意: 发现 {aa_counts['X']} 个未知氨基酸 (X)，占比 {aa_counts['X']/total_aa*100:.2f}%")

print("\n氨基酸分类统计:")
hydrophobic = "AILMFWV"
hydrophilic = "STYNQ"
acidic = "DE"
basic = "KRH"
aromatic = "FYW"
cysteine = "C"
glycine = "G"
proline = "P"

hydrophobic_count = sum(aa_counts[aa] for aa in hydrophobic)
hydrophilic_count = sum(aa_counts[aa] for aa in hydrophilic)
acidic_count = sum(aa_counts[aa] for aa in acidic)
basic_count = sum(aa_counts[aa] for aa in basic)
aromatic_count = sum(aa_counts[aa] for aa in aromatic)
cysteine_count = aa_counts["C"]
glycine_count = aa_counts["G"]
proline_count = aa_counts["P"]

print(f"  疏水性氨基酸 ({hydrophobic}): {hydrophobic_count:>6} ({hydrophobic_count/total_aa*100:>5.2f}%)")
print(f"  亲水性氨基酸 ({hydrophilic}): {hydrophilic_count:>6} ({hydrophilic_count/total_aa*100:>5.2f}%)")
print(f"  酸性氨基酸 ({acidic}): {acidic_count:>6} ({acidic_count/total_aa*100:>5.2f}%)")
print(f"  碱性氨基酸 ({basic}): {basic_count:>6} ({basic_count/total_aa*100:>5.2f}%)")
print(f"  芳香族氨基酸 ({aromatic}): {aromatic_count:>6} ({aromatic_count/total_aa*100:>5.2f}%)")
print(f"  半胱氨酸 (C): {cysteine_count:>6} ({cysteine_count/total_aa*100:>5.2f}%)")
print(f"  甘氨酸 (G): {glycine_count:>6} ({glycine_count/total_aa*100:>5.2f}%)")
print(f"  脯氨酸 (P): {proline_count:>6} ({proline_count/total_aa*100:>5.2f}%)")

print("\n特殊序列特征:")
cysteine_containing = sum(1 for seq in sequences if "C" in seq)
print(f"  含半胱氨酸的序列: {cysteine_containing} ({cysteine_containing/len(sequences)*100:.1f}%)")

disulfide_bonds = sum(seq.count("C") // 2 for seq in sequences if "C" in seq)
print(f"  可能的二硫键数量: {disulfide_bonds}")

glycine_rich = sum(1 for seq in sequences if seq.count("G") / len(seq) > 0.15)
print(f"  富含甘氨酸的序列 (>15%): {glycine_rich} ({glycine_rich/len(sequences)*100:.1f}%)")

charged_sequences = sum(1 for seq in sequences if any(aa in seq for aa in "DEKRH"))
print(f"  含带电氨基酸的序列: {charged_sequences} ({charged_sequences/len(sequences)*100:.1f}%)")

x_containing = sum(1 for seq in sequences if "X" in seq)
if x_containing > 0:
    print(f"  含未知氨基酸 (X) 的序列: {x_containing} ({x_containing/len(sequences)*100:.1f}%)")

print("\n序列复杂度分析:")
unique_sequences = len(set(sequences))
print(f"  唯一序列数: {unique_sequences}")
print(f"  序列多样性: {unique_sequences/len(sequences)*100:.1f}%")

most_common_seqs = Counter(sequences).most_common(5)
print(f"\n最常见的5个序列:")
for i, (seq, count) in enumerate(most_common_seqs, 1):
    print(f"  {i}. 长度={len(seq)}: 出现{count}次")
    if len(seq) <= 30:
        print(f"     序列: {seq}")
    else:
        print(f"     序列: {seq[:30]}...")

print("\n序列起始和结尾氨基酸:")
first_aa = Counter(seq[0] for seq in sequences if seq)
last_aa = Counter(seq[-1] for seq in sequences if seq)

print("  起始氨基酸频率 (前5):")
for aa, count in first_aa.most_common(5):
    percentage = count / len(sequences) * 100
    print(f"    {aa}: {count:>4} ({percentage:>5.1f}%)")

print("  结尾氨基酸频率 (前5):")
for aa, count in last_aa.most_common(5):
    percentage = count / len(sequences) * 100
    print(f"    {aa}: {count:>4} ({percentage:>5.1f}%)")

print("\n序列模式分析:")
patterns = {
    "C...C": sum(1 for seq in sequences if len(seq) >= 4 and seq[0] == "C" and seq[-1] == "C"),
    "G...G": sum(1 for seq in sequences if len(seq) >= 4 and seq[0] == "G" and seq[-1] == "G"),
    "K...K": sum(1 for seq in sequences if len(seq) >= 4 and seq[0] == "K" and seq[-1] == "K"),
    "R...R": sum(1 for seq in sequences if len(seq) >= 4 and seq[0] == "R" and seq[-1] == "R"),
}

for pattern, count in patterns.items():
    percentage = count / len(sequences) * 100
    if count > 0:
        print(f"  {pattern:>6}: {count:>4} 条 ({percentage:>5.1f}%)")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(f"代表性序列集包含 {len(sequences)} 条肽序列")
print(f"平均长度 {np.mean(lengths):.2f} aa")
print(f"序列多样性高，{unique_sequences} 条唯一序列")
print(f"富含疏水性氨基酸 ({hydrophobic_count/total_aa*100:.1f}%)")
print(f"{cysteine_containing/len(sequences)*100:.1f}% 的序列含半胱氨酸，可能形成二硫键")

if 'X' in aa_counts:
    print(f"注意: {aa_counts['X']/total_aa*100:.1f}% 的氨基酸为未知类型 (X)")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("代表性序列特征分析报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"总序列数: {len(sequences)}\n\n")
    f.write("序列长度统计:\n")
    f.write(f"  最小长度: {min(lengths)} aa\n")
    f.write(f"  最大长度: {max(lengths)} aa\n")
    f.write(f"  平均长度: {np.mean(lengths):.2f} aa\n")
    f.write(f"  中位数长度: {np.median(lengths):.2f} aa\n")
    f.write(f"  标准差: {np.std(lengths):.2f} aa\n\n")
    f.write("序列长度分布:\n")
    for (min_len, max_len), label in zip(length_ranges, range_labels):
        count = sum(1 for length in lengths if min_len <= length <= max_len)
        percentage = count / len(lengths) * 100
        if count > 0:
            f.write(f"  {label:>8}: {count:>4} 条 ({percentage:>5.1f}%)\n")
    f.write("\n氨基酸组成分析:\n")
    f.write(f"  总氨基酸数: {total_aa}\n")
    f.write(f"  氨基酸种类: {len(aa_counts)}\n\n")
    f.write("各氨基酸出现频率:\n")
    for aa, count in aa_counts.most_common():
        percentage = count / total_aa * 100
        f.write(f"  {aa}: {count} ({percentage:.2f}%)\n")
    f.write("\n氨基酸分类统计:\n")
    f.write(f"  疏水性氨基酸 ({hydrophobic}): {hydrophobic_count} ({hydrophobic_count/total_aa*100:.2f}%)\n")
    f.write(f"  亲水性氨基酸 ({hydrophilic}): {hydrophilic_count} ({hydrophilic_count/total_aa*100:.2f}%)\n")
    f.write(f"  酸性氨基酸 ({acidic}): {acidic_count} ({acidic_count/total_aa*100:.2f}%)\n")
    f.write(f"  碱性氨基酸 ({basic}): {basic_count} ({basic_count/total_aa*100:.2f}%)\n")
    f.write(f"  芳香族氨基酸 ({aromatic}): {aromatic_count} ({aromatic_count/total_aa*100:.2f}%)\n")
    f.write(f"  半胱氨酸 (C): {cysteine_count} ({cysteine_count/total_aa*100:.2f}%)\n")
    f.write(f"  甘氨酸 (G): {glycine_count} ({glycine_count/total_aa*100:.2f}%)\n")
    f.write(f"  脯氨酸 (P): {proline_count} ({proline_count/total_aa*100:.2f}%)\n")
    if 'X' in aa_counts:
        f.write(f"\n注意: 发现 {aa_counts['X']} 个未知氨基酸 (X)，占比 {aa_counts['X']/total_aa*100:.2f}%\n")

print(f"\n分析结果已保存到: {output_file}")
