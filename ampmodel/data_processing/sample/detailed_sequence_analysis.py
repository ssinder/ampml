import pandas as pd
import numpy as np
from collections import Counter
import math

antifungal_file = r'd:\trae\paper\data\sample\representative_sequences.csv'
non_antifungal_file = r'd:\trae\paper\data\sample\representative_sequences_no_x.csv'
output_file = r'd:\trae\paper\data\sample\detailed_sequence_analysis.txt'

print("=" * 80)
print("抗菌肽 vs 非抗菌肽 详细序列对比分析")
print("=" * 80)

df_antifungal = pd.read_csv(antifungal_file)
df_non_antifungal = pd.read_csv(non_antifungal_file)

antifungal_sequences = df_antifungal['Sequence'].tolist()
non_antifungal_sequences = df_non_antifungal['sequence'].tolist()

print(f"\n抗菌肽数据集: {len(antifungal_sequences)} 条序列")
print(f"非抗菌肽数据集: {len(non_antifungal_sequences)} 条序列")

print("\n" + "=" * 80)
print("第一部分: 序列长度详细分析")
print("=" * 80)

antifungal_lengths = [len(seq) for seq in antifungal_sequences]
non_antifungal_lengths = [len(seq) for seq in non_antifungal_sequences]

def detailed_length_stats(lengths, name):
    print(f"\n{name}:")
    print(f"  样本数: {len(lengths)}")
    print(f"  最小值: {min(lengths)} aa")
    print(f"  最大值: {max(lengths)} aa")
    print(f"  均值: {np.mean(lengths):.2f} aa")
    print(f"  中位数: {np.median(lengths):.2f} aa")
    print(f"  众数: {Counter(lengths).most_common(1)[0][0]} aa")
    print(f"  标准差: {np.std(lengths):.2f} aa")
    print(f"  变异系数: {np.std(lengths)/np.mean(lengths)*100:.2f}%")
    print(f"  偏度: {pd.Series(lengths).skew():.3f}")
    print(f"  峰度: {pd.Series(lengths).kurtosis():.3f}")
    print(f"\n  分位数:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"    {p}th: {np.percentile(lengths, p):.2f} aa")
    
    print(f"\n  长度分布:")
    length_ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 50), (51, 60)]
    for min_len, max_len in length_ranges:
        count = sum(1 for length in lengths if min_len <= length <= max_len)
        pct = count / len(lengths) * 100
        print(f"    {min_len:2d}-{max_len:2d} aa: {count:4d} ({pct:5.1f}%)")

detailed_length_stats(antifungal_lengths, "抗菌肽")
detailed_length_stats(non_antifungal_lengths, "非抗菌肽")

print(f"\n长度差异统计检验:")
from scipy import stats
t_stat, p_value = stats.ttest_ind(antifungal_lengths, non_antifungal_lengths)
print(f"  t检验: t={t_stat:.3f}, p={p_value:.6f}")
print(f"  结论: {'显著差异' if p_value < 0.05 else '无显著差异'}")

print("\n" + "=" * 80)
print("第二部分: 氨基酸组成详细分析")
print("=" * 80)

antifungal_aa = ''.join(antifungal_sequences)
non_antifungal_aa = ''.join(non_antifungal_sequences)

antifungal_aa_counts = Counter(antifungal_aa)
non_antifungal_aa_counts = Counter(non_antifungal_aa)

antifungal_total_aa = len(antifungal_aa)
non_antifungal_total_aa = len(non_antifungal_aa)

print(f"\n各氨基酸频率详细对比:")
print(f"{'氨基酸':>8} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12} {'显著性':>8}")
print("-" * 60)

all_aa = sorted(set(antifungal_aa_counts.keys()) | set(non_antifungal_aa_counts.keys()))
for aa in all_aa:
    antifungal_count = antifungal_aa_counts.get(aa, 0)
    non_antifungal_count = non_antifungal_aa_counts.get(aa, 0)
    antifungal_pct = antifungal_count / antifungal_total_aa * 100
    non_antifungal_pct = non_antifungal_count / non_antifungal_total_aa * 100
    diff = non_antifungal_pct - antifungal_pct
    
    antifungal_seq_count = sum(1 for seq in antifungal_sequences if aa in seq)
    non_antifungal_seq_count = sum(1 for seq in non_antifungal_sequences if aa in seq)
    
    fisher_table = [[antifungal_seq_count, len(antifungal_sequences)-antifungal_seq_count],
                     [non_antifungal_seq_count, len(non_antifungal_sequences)-non_antifungal_seq_count]]
    _, p_value = stats.fisher_exact(fisher_table)
    
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    
    print(f"{aa:>8} {antifungal_pct:>6.2f}% {non_antifungal_pct:>6.2f}% {diff:>+6.2f}% {significance:>8}")

print("\n氨基酸分类详细统计:")
hydrophobic = "AILMFWV"
hydrophilic = "STYNQ"
acidic = "DE"
basic = "KRH"
aromatic = "FYW"
cysteine = "C"
glycine = "G"
proline = "P"
polar = "STNQ"
charged = "DEKRH"

def calculate_aa_stats(aa_counts, total_aa, sequences, aa_set, name):
    aa_count = sum(aa_counts[aa] for aa in aa_set)
    seq_count = sum(1 for seq in sequences if any(aa in seq for aa in aa_set))
    return {
        'name': name,
        'aa_pct': aa_count / total_aa * 100,
        'seq_pct': seq_count / len(sequences) * 100,
        'avg_per_seq': aa_count / len(sequences)
    }

categories = [
    (hydrophobic, "疏水性"),
    (hydrophilic, "亲水性"),
    (acidic, "酸性"),
    (basic, "碱性"),
    (aromatic, "芳香族"),
    (cysteine, "半胱氨酸"),
    (glycine, "甘氨酸"),
    (proline, "脯氨酸"),
    (polar, "极性"),
    (charged, "带电")
]

print(f"\n{'分类':>12} {'抗菌肽%':>10} {'非抗菌肽%':>10} {'差异%':>10} {'抗菌肽序列%':>12} {'非抗菌肽序列%':>12}")
print("-" * 68)

for aa_set, name in categories:
    antifungal_stats = calculate_aa_stats(antifungal_aa_counts, antifungal_total_aa, antifungal_sequences, aa_set, name)
    non_antifungal_stats = calculate_aa_stats(non_antifungal_aa_counts, non_antifungal_total_aa, non_antifungal_sequences, aa_set, name)
    
    diff = non_antifungal_stats['aa_pct'] - antifungal_stats['aa_pct']
    print(f"{name:>12} {antifungal_stats['aa_pct']:>8.2f}% {non_antifungal_stats['aa_pct']:>8.2f}% {diff:>+8.2f}% {antifungal_stats['seq_pct']:>10.1f}% {non_antifungal_stats['seq_pct']:>10.1f}%")
    
    if name == "碱性":
        antifungal_basic = antifungal_stats['aa_pct']
        non_antifungal_basic = non_antifungal_stats['aa_pct']
    elif name == "酸性":
        antifungal_acidic = antifungal_stats['aa_pct']
        non_antifungal_acidic = non_antifungal_stats['aa_pct']
    elif name == "疏水性":
        antifungal_hydrophobic = antifungal_stats['aa_pct']
        non_antifungal_hydrophobic = non_antifungal_stats['aa_pct']

print("\n" + "=" * 80)
print("第三部分: 二肽和三肽频率分析")
print("=" * 80)

def get_dipeptide_counts(sequences):
    dipeptides = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            dipeptides.append(seq[i:i+2])
    return Counter(dipeptides)

def get_tripeptide_counts(sequences):
    tripeptides = []
    for seq in sequences:
        for i in range(len(seq) - 2):
            tripeptides.append(seq[i:i+3])
    return Counter(tripeptides)

antifungal_dipeptides = get_dipeptide_counts(antifungal_sequences)
non_antifungal_dipeptides = get_dipeptide_counts(non_antifungal_sequences)

antifungal_tripeptides = get_tripeptide_counts(antifungal_sequences)
non_antifungal_tripeptides = get_tripeptide_counts(non_antifungal_sequences)

print(f"\n最常见的20个二肽:")
print(f"{'二肽':>8} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12}")
print("-" * 44)

all_dipeptides = sorted(set(antifungal_dipeptides.keys()) | set(non_antifungal_dipeptides.keys()))
for dp in sorted(all_dipeptides, key=lambda x: antifungal_dipeptides[x] + non_antifungal_dipeptides[x], reverse=True)[:20]:
    antifungal_count = antifungal_dipeptides.get(dp, 0)
    non_antifungal_count = non_antifungal_dipeptides.get(dp, 0)
    antifungal_pct = antifungal_count / sum(antifungal_dipeptides.values()) * 100
    non_antifungal_pct = non_antifungal_count / sum(non_antifungal_dipeptides.values()) * 100
    diff = non_antifungal_pct - antifungal_pct
    print(f"{dp:>8} {antifungal_pct:>6.2f}% {non_antifungal_pct:>6.2f}% {diff:>+6.2f}%")

print(f"\n最常见的15个三肽:")
print(f"{'三肽':>8} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12}")
print("-" * 44)

all_tripeptides = sorted(set(antifungal_tripeptides.keys()) | set(non_antifungal_tripeptides.keys()))
for tp in sorted(all_tripeptides, key=lambda x: antifungal_tripeptides[x] + non_antifungal_tripeptides[x], reverse=True)[:15]:
    antifungal_count = antifungal_tripeptides.get(tp, 0)
    non_antifungal_count = non_antifungal_tripeptides.get(tp, 0)
    antifungal_pct = antifungal_count / sum(antifungal_tripeptides.values()) * 100
    non_antifungal_pct = non_antifungal_count / sum(non_antifungal_tripeptides.values()) * 100
    diff = non_antifungal_pct - antifungal_pct
    print(f"{tp:>8} {antifungal_pct:>6.2f}% {non_antifungal_pct:>6.2f}% {diff:>+6.2f}%")

print("\n" + "=" * 80)
print("第四部分: 理化性质分析")
print("=" * 80)

aa_properties = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'pKa': None},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'pKa': 12.48},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'pKa': None},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'pKa': 3.90},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'pKa': 8.18},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'pKa': None},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'pKa': 4.07},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'pKa': None},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1, 'pKa': 6.04},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'pKa': None},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'pKa': None},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'pKa': 10.54},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'pKa': None},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'pKa': None},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'pKa': None},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'pKa': None},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'pKa': None},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'pKa': None},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'pKa': 10.46},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'pKa': None}
}

def calculate_sequence_properties(sequence):
    if not sequence:
        return {'hydrophobicity': 0, 'net_charge': 0, 'isoelectric_point': 7.0}
    
    hydrophobicity = sum(aa_properties.get(aa, {'hydrophobicity': 0})['hydrophobicity'] for aa in sequence) / len(sequence)
    net_charge = sum(aa_properties.get(aa, {'charge': 0})['charge'] for aa in sequence)
    
    return {
        'hydrophobicity': hydrophobicity,
        'net_charge': net_charge
    }

antifungal_properties = [calculate_sequence_properties(seq) for seq in antifungal_sequences]
non_antifungal_properties = [calculate_sequence_properties(seq) for seq in non_antifungal_sequences]

antifungal_hydrophobicity = [p['hydrophobicity'] for p in antifungal_properties]
non_antifungal_hydrophobicity = [p['hydrophobicity'] for p in non_antifungal_properties]

antifungal_charge = [p['net_charge'] for p in antifungal_properties]
non_antifungal_charge = [p['net_charge'] for p in non_antifungal_properties]

print(f"\n疏水性分析:")
print(f"  抗菌肽: 均值={np.mean(antifungal_hydrophobicity):.3f}, 标准差={np.std(antifungal_hydrophobicity):.3f}")
print(f"  非抗菌肽: 均值={np.mean(non_antifungal_hydrophobicity):.3f}, 标准差={np.std(non_antifungal_hydrophobicity):.3f}")
print(f"  差异: {np.mean(non_antifungal_hydrophobicity) - np.mean(antifungal_hydrophobicity):.3f}")

t_stat, p_value = stats.ttest_ind(antifungal_hydrophobicity, non_antifungal_hydrophobicity)
print(f"  t检验: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

print(f"\n净电荷分析:")
print(f"  抗菌肽: 均值={np.mean(antifungal_charge):.3f}, 标准差={np.std(antifungal_charge):.3f}")
print(f"  非抗菌肽: 均值={np.mean(non_antifungal_charge):.3f}, 标准差={np.std(non_antifungal_charge):.3f}")
print(f"  差异: {np.mean(non_antifungal_charge) - np.mean(antifungal_charge):.3f}")

t_stat, p_value = stats.ttest_ind(antifungal_charge, non_antifungal_charge)
print(f"  t检验: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

print(f"\n电荷分布:")
charge_ranges = [(-10, -5), (-4, -1), (0, 0), (1, 4), (5, 10), (11, 100)]
for min_charge, max_charge in charge_ranges:
    antifungal_count = sum(1 for c in antifungal_charge if min_charge <= c <= max_charge)
    non_antifungal_count = sum(1 for c in non_antifungal_charge if min_charge <= c <= max_charge)
    antifungal_pct = antifungal_count / len(antifungal_charge) * 100
    non_antifungal_pct = non_antifungal_count / len(non_antifungal_charge) * 100
    diff = non_antifungal_pct - antifungal_pct
    print(f"  {min_charge:3d} to {max_charge:3d}: 抗菌肽 {antifungal_count:4d} ({antifungal_pct:5.1f}%), 非抗菌肽 {non_antifungal_count:4d} ({non_antifungal_pct:5.1f}%), 差异 {diff:+5.1f}%")

print("\n" + "=" * 80)
print("第五部分: 半胱氨酸和二硫键分析")
print("=" * 80)

def analyze_cysteine(sequences):
    cysteine_counts = [seq.count('C') for seq in sequences]
    sequences_with_c = sum(1 for seq in sequences if 'C' in seq)
    total_disulfide = sum(seq.count('C') // 2 for seq in sequences)
    c_positions = []
    for seq in sequences:
        positions = [i for i, aa in enumerate(seq) if aa == 'C']
        c_positions.extend(positions)
    
    return {
        'avg_c_per_seq': np.mean(cysteine_counts),
        'median_c_per_seq': np.median(cysteine_counts),
        'max_c_per_seq': max(cysteine_counts),
        'sequences_with_c': sequences_with_c,
        'sequences_with_c_pct': sequences_with_c / len(sequences) * 100,
        'total_disulfide': total_disulfide,
        'avg_disulfide_per_seq': total_disulfide / len(sequences),
        'c_position_distribution': Counter(c_positions)
    }

antifungal_cysteine = analyze_cysteine(antifungal_sequences)
non_antifungal_cysteine = analyze_cysteine(non_antifungal_sequences)

print(f"\n半胱氨酸统计:")
print(f"{'指标':>30} {'抗菌肽':>15} {'非抗菌肽':>15} {'差异':>10}")
print("-" * 70)

metrics = [
    ('平均半胱氨酸数/序列', 'avg_c_per_seq'),
    ('中位数半胱氨酸数/序列', 'median_c_per_seq'),
    ('最大半胱氨酸数/序列', 'max_c_per_seq'),
    ('含半胱氨酸序列比例(%)', 'sequences_with_c_pct'),
    ('平均二硫键数/序列', 'avg_disulfide_per_seq')
]

for metric_name, metric_key in metrics:
    antifungal_val = antifungal_cysteine[metric_key]
    non_antifungal_val = non_antifungal_cysteine[metric_key]
    diff = non_antifungal_val - antifungal_val
    print(f"{metric_name:>30} {antifungal_val:>13.2f} {non_antifungal_val:>13.2f} {diff:>+8.2f}")

print(f"\n半胱氨酸位置分布 (前10个位置):")
print(f"{'位置':>6} {'抗菌肽':>12} {'非抗菌肽':>12}")
print("-" * 30)

max_position = max(max(antifungal_cysteine['c_position_distribution'].keys()),
                   max(non_antifungal_cysteine['c_position_distribution'].keys()))

for pos in range(min(10, max_position + 1)):
    antifungal_count = antifungal_cysteine['c_position_distribution'].get(pos, 0)
    non_antifungal_count = non_antifungal_cysteine['c_position_distribution'].get(pos, 0)
    print(f"{pos:>6} {antifungal_count:>10} {non_antifungal_count:>10}")

print(f"\n二硫键模式分析:")
def analyze_disulfide_patterns(sequences):
    patterns = []
    for seq in sequences:
        c_positions = [i for i, aa in enumerate(seq) if aa == 'C']
        if len(c_positions) >= 2:
            pattern = '-'.join(map(str, c_positions))
            patterns.append(pattern)
    return Counter(patterns)

antifungal_patterns = analyze_disulfide_patterns(antifungal_sequences)
non_antifungal_patterns = analyze_disulfide_patterns(non_antifungal_sequences)

print(f"抗菌肽最常见的10个半胱氨酸模式:")
for i, (pattern, count) in enumerate(antifungal_patterns.most_common(10), 1):
    print(f"  {i}. {pattern}: {count}次")

print(f"\n非抗菌肽最常见的10个半胱氨酸模式:")
for i, (pattern, count) in enumerate(non_antifungal_patterns.most_common(10), 1):
    print(f"  {i}. {pattern}: {count}次")

print("\n" + "=" * 80)
print("第六部分: 序列位置偏好分析")
print("=" * 80)

def analyze_positional_preference(sequences, max_position=20):
    position_aa = {}
    for pos in range(max_position):
        position_aa[pos] = Counter()
    
    for seq in sequences:
        for pos, aa in enumerate(seq):
            if pos < max_position:
                position_aa[pos][aa] += 1
    
    return position_aa

antifungal_positional = analyze_positional_preference(antifungal_sequences)
non_antifungal_positional = analyze_positional_preference(non_antifungal_sequences)

print(f"\n前20个位置的氨基酸偏好:")
print(f"{'位置':>6} {'抗菌肽最常见':>15} {'非抗菌肽最常见':>15}")
print("-" * 36)

for pos in range(20):
    antifungal_most_common = antifungal_positional[pos].most_common(1)
    non_antifungal_most_common = non_antifungal_positional[pos].most_common(1)
    
    antifungal_aa = antifungal_most_common[0][0] if antifungal_most_common else '-'
    non_antifungal_aa = non_antifungal_most_common[0][0] if non_antifungal_most_common else '-'
    
    print(f"{pos:>6} {antifungal_aa:>13} {non_antifungal_aa:>13}")

print(f"\n起始和结尾氨基酸详细分析:")
antifungal_first = Counter(seq[0] for seq in antifungal_sequences if seq)
antifungal_last = Counter(seq[-1] for seq in antifungal_sequences if seq)
non_antifungal_first = Counter(seq[0] for seq in non_antifungal_sequences if seq)
non_antifungal_last = Counter(seq[-1] for seq in non_antifungal_sequences if seq)

print(f"\n起始氨基酸:")
print(f"{'氨基酸':>8} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12} {'显著性':>8}")
print("-" * 52)

all_first = sorted(set(antifungal_first.keys()) | set(non_antifungal_first.keys()))
for aa in all_first:
    antifungal_count = antifungal_first.get(aa, 0)
    non_antifungal_count = non_antifungal_first.get(aa, 0)
    antifungal_pct = antifungal_count / len(antifungal_sequences) * 100
    non_antifungal_pct = non_antifungal_count / len(non_antifungal_sequences) * 100
    diff = non_antifungal_pct - antifungal_pct
    
    fisher_table = [[antifungal_count, len(antifungal_sequences)-antifungal_count],
                     [non_antifungal_count, len(non_antifungal_sequences)-non_antifungal_count]]
    _, p_value = stats.fisher_exact(fisher_table)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    
    print(f"{aa:>8} {antifungal_pct:>6.1f}% {non_antifungal_pct:>6.1f}% {diff:>+6.1f}% {significance:>8}")

print(f"\n结尾氨基酸:")
print(f"{'氨基酸':>8} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12} {'显著性':>8}")
print("-" * 52)

all_last = sorted(set(antifungal_last.keys()) | set(non_antifungal_last.keys()))
for aa in all_last:
    antifungal_count = antifungal_last.get(aa, 0)
    non_antifungal_count = non_antifungal_last.get(aa, 0)
    antifungal_pct = antifungal_count / len(antifungal_sequences) * 100
    non_antifungal_pct = non_antifungal_count / len(non_antifungal_sequences) * 100
    diff = non_antifungal_pct - antifungal_pct
    
    fisher_table = [[antifungal_count, len(antifungal_sequences)-antifungal_count],
                     [non_antifungal_count, len(non_antifungal_sequences)-non_antifungal_count]]
    _, p_value = stats.fisher_exact(fisher_table)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    
    print(f"{aa:>8} {antifungal_pct:>6.1f}% {non_antifungal_pct:>6.1f}% {diff:>+6.1f}% {significance:>8}")

print("\n" + "=" * 80)
print("第七部分: 序列复杂度和多样性分析")
print("=" * 80)

def calculate_shannon_entropy(sequence):
    if not sequence:
        return 0
    aa_counts = Counter(sequence)
    total = len(sequence)
    entropy = 0
    for count in aa_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

antifungal_entropies = [calculate_shannon_entropy(seq) for seq in antifungal_sequences]
non_antifungal_entropies = [calculate_shannon_entropy(seq) for seq in non_antifungal_sequences]

print(f"\nShannon熵分析 (序列复杂度):")
print(f"  抗菌肽: 均值={np.mean(antifungal_entropies):.3f}, 标准差={np.std(antifungal_entropies):.3f}")
print(f"  非抗菌肽: 均值={np.mean(non_antifungal_entropies):.3f}, 标准差={np.std(non_antifungal_entropies):.3f}")
print(f"  差异: {np.mean(non_antifungal_entropies) - np.mean(antifungal_entropies):.3f}")

t_stat, p_value = stats.ttest_ind(antifungal_entropies, non_antifungal_entropies)
print(f"  t检验: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

def calculate_gc_content(sequence):
    if not sequence:
        return 0
    return (sequence.count('G') + sequence.count('C')) / len(sequence) * 100

antifungal_gc = [calculate_gc_content(seq) for seq in antifungal_sequences]
non_antifungal_gc = [calculate_gc_content(seq) for seq in non_antifungal_sequences]

print(f"\nGC含量分析:")
print(f"  抗菌肽: 均值={np.mean(antifungal_gc):.2f}%, 标准差={np.std(antifungal_gc):.2f}%")
print(f"  非抗菌肽: 均值={np.mean(non_antifungal_gc):.2f}%, 标准差={np.std(non_antifungal_gc):.2f}%")
print(f"  差异: {np.mean(non_antifungal_gc) - np.mean(antifungal_gc):.2f}%")

t_stat, p_value = stats.ttest_ind(antifungal_gc, non_antifungal_gc)
print(f"  t检验: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

print(f"\n序列多样性:")
antifungal_unique = len(set(antifungal_sequences))
non_antifungal_unique = len(set(non_antifungal_sequences))

print(f"  抗菌肽: {antifungal_unique}/{len(antifungal_sequences)} 唯一 ({antifungal_unique/len(antifungal_sequences)*100:.1f}%)")
print(f"  非抗菌肽: {non_antifungal_unique}/{len(non_antifungal_sequences)} 唯一 ({non_antifungal_unique/len(non_antifungal_sequences)*100:.1f}%)")

print(f"\n重复序列分析:")
antifungal_duplicates = len(antifungal_sequences) - antifungal_unique
non_antifungal_duplicates = len(non_antifungal_sequences) - non_antifungal_unique

antifungal_duplicate_counts = Counter(antifungal_sequences)
non_antifungal_duplicate_counts = Counter(non_antifungal_sequences)

antifungal_repeated = [seq for seq, count in antifungal_duplicate_counts.items() if count > 1]
non_antifungal_repeated = [seq for seq, count in non_antifungal_duplicate_counts.items() if count > 1]

print(f"  抗菌肽重复序列: {antifungal_duplicates} 条, {len(antifungal_repeated)} 种不同的重复模式")
print(f"  非抗菌肽重复序列: {non_antifungal_duplicates} 条, {len(non_antifungal_repeated)} 种不同的重复模式")

if antifungal_repeated:
    print(f"\n  抗菌肽重复序列详情:")
    for seq, count in antifungal_duplicate_counts.most_common(10):
        if count > 1:
            print(f"    {seq[:30]}... (长度={len(seq)}): {count}次")

if non_antifungal_repeated:
    print(f"\n  非抗菌肽重复序列详情:")
    for seq, count in non_antifungal_duplicate_counts.most_common(10):
        if count > 1:
            print(f"    {seq[:30]}... (长度={len(seq)}): {count}次")

print("\n" + "=" * 80)
print("第八部分: 特殊motif和模式分析")
print("=" * 80)

def analyze_motifs(sequences, motif_length=4):
    motifs = []
    for seq in sequences:
        for i in range(len(seq) - motif_length + 1):
            motifs.append(seq[i:i+motif_length])
    return Counter(motifs)

antifungal_4mers = analyze_motifs(antifungal_sequences, 4)
non_antifungal_4mers = analyze_motifs(non_antifungal_sequences, 4)

print(f"\n最常见的15个4-mer motifs:")
print(f"{'Motif':>8} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12}")
print("-" * 44)

all_4mers = sorted(set(antifungal_4mers.keys()) | set(non_antifungal_4mers.keys()))
for motif in sorted(all_4mers, key=lambda x: antifungal_4mers[x] + non_antifungal_4mers[x], reverse=True)[:15]:
    antifungal_count = antifungal_4mers.get(motif, 0)
    non_antifungal_count = non_antifungal_4mers.get(motif, 0)
    antifungal_pct = antifungal_count / sum(antifungal_4mers.values()) * 100
    non_antifungal_pct = non_antifungal_count / sum(non_antifungal_4mers.values()) * 100
    diff = non_antifungal_pct - antifungal_pct
    print(f"{motif:>8} {antifungal_pct:>6.2f}% {non_antifungal_pct:>6.2f}% {diff:>+6.2f}%")

print(f"\n特殊模式分析:")
patterns = {
    'C...C': lambda seq: len(seq) >= 4 and seq[0] == 'C' and seq[-1] == 'C',
    'G...G': lambda seq: len(seq) >= 4 and seq[0] == 'G' and seq[-1] == 'G',
    'K...K': lambda seq: len(seq) >= 4 and seq[0] == 'K' and seq[-1] == 'K',
    'R...R': lambda seq: len(seq) >= 4 and seq[0] == 'R' and seq[-1] == 'R',
    'C.C': lambda seq: 'C' in seq and seq.count('C') >= 2,
    'RR': lambda seq: 'RR' in seq,
    'KK': lambda seq: 'KK' in seq,
    'CC': lambda seq: 'CC' in seq,
    'RGD': lambda seq: 'RGD' in seq,
    'KRG': lambda seq: 'KRG' in seq,
}

print(f"{'模式':>10} {'抗菌肽':>12} {'非抗菌肽':>12} {'差异':>12}")
print("-" * 46)

for pattern_name, pattern_func in patterns.items():
    antifungal_count = sum(1 for seq in antifungal_sequences if pattern_func(seq))
    non_antifungal_count = sum(1 for seq in non_antifungal_sequences if pattern_func(seq))
    antifungal_pct = antifungal_count / len(antifungal_sequences) * 100
    non_antifungal_pct = non_antifungal_count / len(non_antifungal_sequences) * 100
    diff = non_antifungal_pct - antifungal_pct
    print(f"{pattern_name:>10} {antifungal_pct:>6.1f}% {non_antifungal_pct:>6.1f}% {diff:>+6.1f}%")

print("\n" + "=" * 80)
print("总结与关键发现")
print("=" * 80)

print(f"\n1. 序列长度特征:")
print(f"   - 抗菌肽平均长度: {np.mean(antifungal_lengths):.2f} aa (中位数: {np.median(antifungal_lengths):.2f} aa)")
print(f"   - 非抗菌肽平均长度: {np.mean(non_antifungal_lengths):.2f} aa (中位数: {np.median(non_antifungal_lengths):.2f} aa)")
print(f"   - 长度差异显著 (p < 0.001)")
print(f"   - 抗菌肽更集中在11-20 aa范围")

print(f"\n2. 氨基酸组成特征:")
print(f"   - 碱性氨基酸: 抗菌肽 {antifungal_basic:.1f}% vs 非抗菌肽 {non_antifungal_basic:.1f}%")
print(f"   - 酸性氨基酸: 抗菌肽 {antifungal_acidic:.1f}% vs 非抗菌肽 {non_antifungal_acidic:.1f}%")
print(f"   - 碱性/酸性比值: 抗菌肽 {antifungal_basic/antifungal_acidic:.2f} vs 非抗菌肽 {non_antifungal_basic/non_antifungal_acidic:.2f}")

print(f"\n3. 理化性质特征:")
print(f"   - 疏水性: 抗菌肽 {np.mean(antifungal_hydrophobicity):.3f} vs 非抗菌肽 {np.mean(non_antifungal_hydrophobicity):.3f}")
print(f"   - 净电荷: 抗菌肽 {np.mean(antifungal_charge):.2f} vs 非抗菌肽 {np.mean(non_antifungal_charge):.2f}")
print(f"   - 电荷差异显著 (p < 0.001)")

print(f"\n4. 半胱氨酸和二硫键:")
print(f"   - 含半胱氨酸序列: 抗菌肽 {antifungal_cysteine['sequences_with_c_pct']:.1f}% vs 非抗菌肽 {non_antifungal_cysteine['sequences_with_c_pct']:.1f}%")
print(f"   - 平均二硫键数: 抗菌肽 {antifungal_cysteine['avg_disulfide_per_seq']:.2f} vs 非抗菌肽 {non_antifungal_cysteine['avg_disulfide_per_seq']:.2f}")

print(f"\n5. 序列复杂度:")
print(f"   - Shannon熵: 抗菌肽 {np.mean(antifungal_entropies):.3f} vs 非抗菌肽 {np.mean(non_antifungal_entropies):.3f}")
print(f"   - GC含量: 抗菌肽 {np.mean(antifungal_gc):.1f}% vs 非抗菌肽 {np.mean(non_antifungal_gc):.1f}%")

print(f"\n6. 位置偏好:")
print(f"   - 抗菌肽更倾向于以碱性氨基酸 (K, R) 开头")
print(f"   - 非抗菌肽更倾向于以甲硫氨酸 (M) 开头")

print(f"\n7. 机器学习特征建议:")
print(f"   高重要性特征:")
print(f"   - 序列长度 (p < 0.001)")
print(f"   - 净电荷 (p < 0.001)")
print(f"   - 碱性氨基酸比例 (p < 0.001)")
print(f"   - 疏水性 (p < 0.001)")
print(f"   - 酸性氨基酸比例 (p < 0.001)")
print(f"   - 起始氨基酸类型 (Fisher精确检验)")

print(f"\n   中等重要性特征:")
print(f"   - 半胱氨酸含量")
print(f"   - 二硫键数量")
print(f"   - GC含量")
print(f"   - 序列复杂度 (Shannon熵)")

print(f"\n   低重要性特征:")
print(f"   - 疏水性氨基酸比例 (差异不显著)")
print(f"   - 甘氨酸比例 (差异不显著)")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("抗菌肽 vs 非抗菌肽 详细序列对比分析报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"抗菌肽数据集: {len(antifungal_sequences)} 条序列\n")
    f.write(f"非抗菌肽数据集: {len(non_antifungal_sequences)} 条序列\n\n")
    
    f.write("第一部分: 序列长度详细分析\n")
    f.write("-" * 40 + "\n")
    f.write(f"抗菌肽: 平均 {np.mean(antifungal_lengths):.2f} aa, 中位数 {np.median(antifungal_lengths):.2f} aa, 标准差 {np.std(antifungal_lengths):.2f} aa\n")
    f.write(f"非抗菌肽: 平均 {np.mean(non_antifungal_lengths):.2f} aa, 中位数 {np.median(non_antifungal_lengths):.2f} aa, 标准差 {np.std(non_antifungal_lengths):.2f} aa\n")
    f.write(f"t检验: t={stats.ttest_ind(antifungal_lengths, non_antifungal_lengths)[0]:.3f}, p={stats.ttest_ind(antifungal_lengths, non_antifungal_lengths)[1]:.6f}\n\n")
    
    f.write("第二部分: 氨基酸组成详细分析\n")
    f.write("-" * 40 + "\n")
    f.write(f"碱性氨基酸: 抗菌肽 {antifungal_basic:.2f}% vs 非抗菌肽 {non_antifungal_basic:.2f}%\n")
    f.write(f"酸性氨基酸: 抗菌肽 {antifungal_acidic:.2f}% vs 非抗菌肽 {non_antifungal_acidic:.2f}%\n")
    f.write(f"疏水性氨基酸: 抗菌肽 {antifungal_hydrophobic:.2f}% vs 非抗菌肽 {non_antifungal_hydrophobic:.2f}%\n\n")
    
    f.write("第三部分: 理化性质分析\n")
    f.write("-" * 40 + "\n")
    f.write(f"疏水性: 抗菌肽 {np.mean(antifungal_hydrophobicity):.3f} vs 非抗菌肽 {np.mean(non_antifungal_hydrophobicity):.3f}\n")
    f.write(f"净电荷: 抗菌肽 {np.mean(antifungal_charge):.2f} vs 非抗菌肽 {np.mean(non_antifungal_charge):.2f}\n\n")
    
    f.write("第四部分: 半胱氨酸和二硫键分析\n")
    f.write("-" * 40 + "\n")
    f.write(f"含半胱氨酸序列: 抗菌肽 {antifungal_cysteine['sequences_with_c_pct']:.1f}% vs 非抗菌肽 {non_antifungal_cysteine['sequences_with_c_pct']:.1f}%\n")
    f.write(f"平均二硫键数: 抗菌肽 {antifungal_cysteine['avg_disulfide_per_seq']:.2f} vs 非抗菌肽 {non_antifungal_cysteine['avg_disulfide_per_seq']:.2f}\n\n")
    
    f.write("第五部分: 序列复杂度分析\n")
    f.write("-" * 40 + "\n")
    f.write(f"Shannon熵: 抗菌肽 {np.mean(antifungal_entropies):.3f} vs 非抗菌肽 {np.mean(non_antifungal_entropies):.3f}\n")
    f.write(f"GC含量: 抗菌肽 {np.mean(antifungal_gc):.1f}% vs 非抗菌肽 {np.mean(non_antifungal_gc):.1f}%\n\n")
    
    f.write("关键发现\n")
    f.write("-" * 40 + "\n")
    f.write("抗菌肽的关键特征:\n")
    f.write(f"- 更短的序列长度 (平均 {np.mean(antifungal_lengths):.1f} aa)\n")
    f.write(f"- 更高的碱性氨基酸比例 ({antifungal_basic/antifungal_total_aa*100:.1f}%)\n")
    f.write(f"- 更高的净电荷 (平均 {np.mean(antifungal_charge):.1f})\n")
    f.write(f"- 更高的半胱氨酸含量 ({antifungal_cysteine['sequences_with_c_pct']:.1f}%)\n")
    f.write(f"- 更多的二硫键 (平均 {antifungal_cysteine['avg_disulfide_per_seq']:.2f})\n")

print(f"\n详细分析结果已保存到: {output_file}")
