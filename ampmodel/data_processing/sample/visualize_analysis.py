import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

antifungal_file = r'd:\trae\paper\data\sample\representative_sequences.csv'
non_antifungal_file = r'd:\trae\paper\data\sample\representative_sequences_no_x.csv'

df_antifungal = pd.read_csv(antifungal_file)
df_non_antifungal = pd.read_csv(non_antifungal_file)

antifungal_sequences = df_antifungal['Sequence'].tolist()
non_antifungal_sequences = df_non_antifungal['sequence'].tolist()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20, 16))

antifungal_lengths = [len(seq) for seq in antifungal_sequences]
non_antifungal_lengths = [len(seq) for seq in non_antifungal_sequences]

ax1 = plt.subplot(3, 3, 1)
ax1.hist(antifungal_lengths, bins=30, alpha=0.7, label='抗菌肽', color='blue', density=True)
ax1.hist(non_antifungal_lengths, bins=30, alpha=0.7, label='非抗菌肽', color='red', density=True)
ax1.set_xlabel('序列长度 (aa)', fontsize=12)
ax1.set_ylabel('密度', fontsize=12)
ax1.set_title('序列长度分布', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

antifungal_aa = ''.join(antifungal_sequences)
non_antifungal_aa = ''.join(non_antifungal_sequences)

antifungal_aa_counts = Counter(antifungal_aa)
non_antifungal_aa_counts = Counter(non_antifungal_aa)

all_aa = sorted(set(antifungal_aa_counts.keys()) | set(non_antifungal_aa_counts.keys()))
antifungal_aa_freq = [antifungal_aa_counts.get(aa, 0) / len(antifungal_aa) * 100 for aa in all_aa]
non_antifungal_aa_freq = [non_antifungal_aa_counts.get(aa, 0) / len(non_antifungal_aa) * 100 for aa in all_aa]

ax2 = plt.subplot(3, 3, 2)
x = np.arange(len(all_aa))
width = 0.35
ax2.bar(x - width/2, antifungal_aa_freq, width, label='抗菌肽', color='blue', alpha=0.7)
ax2.bar(x + width/2, non_antifungal_aa_freq, width, label='非抗菌肽', color='red', alpha=0.7)
ax2.set_xlabel('氨基酸', fontsize=12)
ax2.set_ylabel('频率 (%)', fontsize=12)
ax2.set_title('氨基酸组成对比', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(all_aa)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

aa_diff = [non_antifungal_aa_counts.get(aa, 0) / len(non_antifungal_aa) * 100 - 
           antifungal_aa_counts.get(aa, 0) / len(antifungal_aa) * 100 for aa in all_aa]

ax3 = plt.subplot(3, 3, 3)
colors = ['green' if d > 0 else 'orange' for d in aa_diff]
ax3.barh(all_aa, aa_diff, color=colors, alpha=0.7)
ax3.set_xlabel('频率差异 (%)', fontsize=12)
ax3.set_ylabel('氨基酸', fontsize=12)
ax3.set_title('氨基酸频率差异 (非抗菌肽 - 抗菌肽)', fontsize=14, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.grid(True, alpha=0.3, axis='x')

hydrophobic = "AILMFWV"
hydrophilic = "STYNQ"
acidic = "DE"
basic = "KRH"
aromatic = "FYW"
cysteine = "C"
glycine = "G"
proline = "P"

categories = [
    ('疏水性', hydrophobic),
    ('亲水性', hydrophilic),
    ('酸性', acidic),
    ('碱性', basic),
    ('芳香族', aromatic),
    ('半胱氨酸', cysteine),
    ('甘氨酸', glycine),
    ('脯氨酸', proline)
]

category_names = []
antifungal_category_freq = []
non_antifungal_category_freq = []

for name, aa_set in categories:
    antifungal_count = sum(antifungal_aa_counts[aa] for aa in aa_set)
    non_antifungal_count = sum(non_antifungal_aa_counts[aa] for aa in aa_set)
    
    category_names.append(name)
    antifungal_category_freq.append(antifungal_count / len(antifungal_aa) * 100)
    non_antifungal_category_freq.append(non_antifungal_count / len(non_antifungal_aa) * 100)

ax4 = plt.subplot(3, 3, 4)
x = np.arange(len(category_names))
width = 0.35
ax4.bar(x - width/2, antifungal_category_freq, width, label='抗菌肽', color='blue', alpha=0.7)
ax4.bar(x + width/2, non_antifungal_category_freq, width, label='非抗菌肽', color='red', alpha=0.7)
ax4.set_xlabel('氨基酸分类', fontsize=12)
ax4.set_ylabel('频率 (%)', fontsize=12)
ax4.set_title('氨基酸分类对比', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(category_names, rotation=45, ha='right')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

aa_properties = {
    'A': {'hydrophobicity': 1.8, 'charge': 0},
    'R': {'hydrophobicity': -4.5, 'charge': 1},
    'N': {'hydrophobicity': -3.5, 'charge': 0},
    'D': {'hydrophobicity': -3.5, 'charge': -1},
    'C': {'hydrophobicity': 2.5, 'charge': 0},
    'Q': {'hydrophobicity': -3.5, 'charge': 0},
    'E': {'hydrophobicity': -3.5, 'charge': -1},
    'G': {'hydrophobicity': -0.4, 'charge': 0},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1},
    'I': {'hydrophobicity': 4.5, 'charge': 0},
    'L': {'hydrophobicity': 3.8, 'charge': 0},
    'K': {'hydrophobicity': -3.9, 'charge': 1},
    'M': {'hydrophobicity': 1.9, 'charge': 0},
    'F': {'hydrophobicity': 2.8, 'charge': 0},
    'P': {'hydrophobicity': -1.6, 'charge': 0},
    'S': {'hydrophobicity': -0.8, 'charge': 0},
    'T': {'hydrophobicity': -0.7, 'charge': 0},
    'W': {'hydrophobicity': -0.9, 'charge': 0},
    'Y': {'hydrophobicity': -1.3, 'charge': 0},
    'V': {'hydrophobicity': 4.2, 'charge': 0}
}

def calculate_sequence_properties(sequence):
    if not sequence:
        return {'hydrophobicity': 0, 'net_charge': 0}
    
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

ax5 = plt.subplot(3, 3, 5)
ax5.hist(antifungal_hydrophobicity, bins=30, alpha=0.7, label='抗菌肽', color='blue', density=True)
ax5.hist(non_antifungal_hydrophobicity, bins=30, alpha=0.7, label='非抗菌肽', color='red', density=True)
ax5.set_xlabel('疏水性', fontsize=12)
ax5.set_ylabel('密度', fontsize=12)
ax5.set_title('疏水性分布', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
ax6.hist(antifungal_charge, bins=30, alpha=0.7, label='抗菌肽', color='blue', density=True)
ax6.hist(non_antifungal_charge, bins=30, alpha=0.7, label='非抗菌肽', color='red', density=True)
ax6.set_xlabel('净电荷', fontsize=12)
ax6.set_ylabel('密度', fontsize=12)
ax6.set_title('净电荷分布', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

antifungal_first = Counter(seq[0] for seq in antifungal_sequences if seq)
non_antifungal_first = Counter(seq[0] for seq in non_antifungal_sequences if seq)

antifungal_last = Counter(seq[-1] for seq in antifungal_sequences if seq)
non_antifungal_last = Counter(seq[-1] for seq in non_antifungal_sequences if seq)

all_first = sorted(set(antifungal_first.keys()) | set(non_antifungal_first.keys()))
antifungal_first_freq = [antifungal_first.get(aa, 0) / len(antifungal_sequences) * 100 for aa in all_first]
non_antifungal_first_freq = [non_antifungal_first.get(aa, 0) / len(non_antifungal_sequences) * 100 for aa in all_first]

ax7 = plt.subplot(3, 3, 7)
x = np.arange(len(all_first))
width = 0.35
ax7.bar(x - width/2, antifungal_first_freq, width, label='抗菌肽', color='blue', alpha=0.7)
ax7.bar(x + width/2, non_antifungal_first_freq, width, label='非抗菌肽', color='red', alpha=0.7)
ax7.set_xlabel('起始氨基酸', fontsize=12)
ax7.set_ylabel('频率 (%)', fontsize=12)
ax7.set_title('起始氨基酸频率', fontsize=14, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(all_first)
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

all_last = sorted(set(antifungal_last.keys()) | set(non_antifungal_last.keys()))
antifungal_last_freq = [antifungal_last.get(aa, 0) / len(antifungal_sequences) * 100 for aa in all_last]
non_antifungal_last_freq = [non_antifungal_last.get(aa, 0) / len(non_antifungal_sequences) * 100 for aa in all_last]

ax8 = plt.subplot(3, 3, 8)
x = np.arange(len(all_last))
width = 0.35
ax8.bar(x - width/2, antifungal_last_freq, width, label='抗菌肽', color='blue', alpha=0.7)
ax8.bar(x + width/2, non_antifungal_last_freq, width, label='非抗菌肽', color='red', alpha=0.7)
ax8.set_xlabel('结尾氨基酸', fontsize=12)
ax8.set_ylabel('频率 (%)', fontsize=12)
ax8.set_title('结尾氨基酸频率', fontsize=14, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(all_last)
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3, axis='y')

import math

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

ax9 = plt.subplot(3, 3, 9)
ax9.hist(antifungal_entropies, bins=30, alpha=0.7, label='抗菌肽', color='blue', density=True)
ax9.hist(non_antifungal_entropies, bins=30, alpha=0.7, label='非抗菌肽', color='red', density=True)
ax9.set_xlabel('Shannon熵', fontsize=12)
ax9.set_ylabel('密度', fontsize=12)
ax9.set_title('序列复杂度 (Shannon熵)', fontsize=14, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'd:\trae\paper\data\sample\sequence_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存到: d:\\trae\\paper\\data\\sample\\sequence_analysis_visualization.png")

fig2 = plt.figure(figsize=(16, 6))

ax1 = plt.subplot(1, 2, 1)
length_ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
range_labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60"]

antifungal_range_counts = []
non_antifungal_range_counts = []

for min_len, max_len in length_ranges:
    antifungal_count = sum(1 for length in antifungal_lengths if min_len <= length <= max_len)
    non_antifungal_count = sum(1 for length in non_antifungal_lengths if min_len <= length <= max_len)
    antifungal_range_counts.append(antifungal_count / len(antifungal_lengths) * 100)
    non_antifungal_range_counts.append(non_antifungal_count / len(non_antifungal_lengths) * 100)

x = np.arange(len(range_labels))
width = 0.35
ax1.bar(x - width/2, antifungal_range_counts, width, label='抗菌肽', color='blue', alpha=0.7)
ax1.bar(x + width/2, non_antifungal_range_counts, width, label='非抗菌肽', color='red', alpha=0.7)
ax1.set_xlabel('序列长度范围 (aa)', fontsize=12)
ax1.set_ylabel('频率 (%)', fontsize=12)
ax1.set_title('序列长度范围分布', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(range_labels)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = plt.subplot(1, 2, 2)
charge_ranges = [(-10, -5), (-4, -1), (0, 0), (1, 4), (5, 10), (11, 100)]
charge_labels = ["-10 to -5", "-4 to -1", "0", "1 to 4", "5 to 10", ">10"]

antifungal_charge_counts = []
non_antifungal_charge_counts = []

for min_charge, max_charge in charge_ranges:
    antifungal_count = sum(1 for c in antifungal_charge if min_charge <= c <= max_charge)
    non_antifungal_count = sum(1 for c in non_antifungal_charge if min_charge <= c <= max_charge)
    antifungal_charge_counts.append(antifungal_count / len(antifungal_charge) * 100)
    non_antifungal_charge_counts.append(non_antifungal_count / len(non_antifungal_charge) * 100)

x = np.arange(len(charge_labels))
width = 0.35
ax2.bar(x - width/2, antifungal_charge_counts, width, label='抗菌肽', color='blue', alpha=0.7)
ax2.bar(x + width/2, non_antifungal_charge_counts, width, label='非抗菌肽', color='red', alpha=0.7)
ax2.set_xlabel('净电荷', fontsize=12)
ax2.set_ylabel('频率 (%)', fontsize=12)
ax2.set_title('净电荷分布', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(charge_labels, rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r'd:\trae\paper\data\sample\sequence_analysis_distribution.png', dpi=300, bbox_inches='tight')
print("分布图已保存到: d:\\trae\\paper\\data\\sample\\sequence_analysis_distribution.png")

plt.show()
