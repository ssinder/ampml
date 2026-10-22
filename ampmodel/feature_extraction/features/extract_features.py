# -*- coding: utf-8 -*-
"""
蛋白质序列特征提取脚本 - 用于两个CSV文件的特征提取
输入: id_sequence_extracted.csv (标签0) + representative_sequences_id_seq.csv (标签1)
输出: 合并后的CSV特征文件
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import io

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data_processing" / "sample"
OUTPUT_DIR = PROJECT_ROOT / "feature_extraction" / "features" / "results"
DOCS_DIR = PROJECT_ROOT / "docs" / "github" / "iFeature-master"

# 输入CSV文件
# 使用 sequences_with_labels.csv 文件
CSV_FILE = OUTPUT_DIR / "sequences_with_labels.csv"

IFeature_SCRIPT = DOCS_DIR / "iFeature.py"
IFeaturePseKRAAC_SCRIPT = DOCS_DIR / "iFeaturePseKRAAC.py"

TEMP_FASTA = OUTPUT_DIR / "temp_sequences.fasta"
OUTPUT_CSV = OUTPUT_DIR / "all_features_combined.csv"

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# 辅助函数：解析FASTA文件
# ============================================================================
def parse_fasta(file_path):
    """解析FASTA文件，返回ID列表和序列列表"""
    sequences = []
    ids = []
    
    with open(file_path, 'r') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append(''.join(current_seq))
                    ids.append(current_id)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences.append(''.join(current_seq))
            ids.append(current_id)
    
    return ids, sequences


# ============================================================================
# 创建合并后的FASTA文件（带标签）
# ============================================================================
def create_combined_fasta():
    """从单个CSV文件创建FASTA文件（过滤掉包含U的序列）"""
    print("=" * 70)
    print("Step 0: Creating FASTA File from sequences_with_labels.csv (Filtering U sequences)")
    print("=" * 70)
    
    # 读取CSV文件
    df = pd.read_csv(CSV_FILE)
    
    print(f"  Total sequences before filtering: {len(df)} sequences")
    
    # 过滤掉包含非标准氨基酸的序列（包括U, B, Z, X, J, O）
    # 标准氨基酸: ACDEFGHIKLMNPQRSTVWY
    non_standard = set('UBZXJWO')
    
    def is_valid_sequence(seq):
        """检查序列是否只包含标准氨基酸且长度>=10"""
        if any(aa in non_standard for aa in seq.upper()):
            return False
        if len(seq) < 10:
            return False
        return True
    
    # 过滤序列
    df_valid = df[df['Sequence'].apply(is_valid_sequence)].copy()
    
    removed = len(df) - len(df_valid)
    
    print(f"  Sequences after filtering: {len(df_valid)} sequences (removed {removed})")
    
    # 写入FASTA文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(TEMP_FASTA, 'w') as f:
        # 写入所有有效序列
        for idx, row in df_valid.iterrows():
            seq_id = row['ID']
            label = row['Label']
            f.write(f">{seq_id} Label:{label}\n")
            f.write(f"{row['Sequence']}\n")
    
    print(f"  FASTA saved to: {TEMP_FASTA}")
    print(f"  Total sequences: {len(df_valid)}")
    
    return str(TEMP_FASTA)


# ============================================================================
# iFeature特征提取
# ============================================================================
def extract_ifeature_features(input_file, output_dir):
    """使用iFeature提取特征（除TPC外）"""
    print("\n" + "=" * 70)
    print("Step 1: Extracting iFeature Features (excluding TPC)")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_types = [
        'AAC', 'DPC', 'CTDC', 'CTDT', 'CTDD', 'GAAC', 'GDPC', 'GTPC',
        'CKSAAP', 'CKSAAGP', 'CTriad', 'KSCTriad', 'DDE',
        'PAAC', 'APAAC', 'Moran', 'Geary', 'NMBroto'
    ]
    
    results = {}
    
    for feature_type in feature_types:
        print(f"\n  Extracting {feature_type}...", end=" ", flush=True)
        
        output_file = os.path.join(output_dir, f"iFeature_{feature_type}.tsv")
        
        try:
            cmd = [
                "python",
                str(IFeature_SCRIPT),
                "--file", input_file,
                "--type", feature_type,
                "--out", output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists(output_file):
                df = pd.read_csv(output_file, sep='\t')
                results[feature_type] = df
                print(f"OK ({df.shape[1] - 1} features)")
            else:
                print("FAIL")
                
        except subprocess.TimeoutExpired:
            print("FAIL: Timeout")
        except Exception as e:
            print(f"FAIL: {e}")
    
    print(f"\n  iFeature extraction completed: {len(results)}/{len(feature_types)} successful")
    return results


# ============================================================================
# PseKRAAC特征提取
# ============================================================================
def extract_psekraac_features(input_file, output_dir):
    """使用PseKRAAC提取伪k元约化氨基酸组成特征"""
    print("\n" + "=" * 70)
    print("Step 2: Extracting PseKRAAC Features")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    psekraac_types = [
        'type1', 'type2', 'type3A', 'type3B', 'type4', 'type5',
        'type6A', 'type6B', 'type6C', 'type7', 'type8', 'type9',
        'type10', 'type11', 'type12', 'type13', 'type14', 'type15', 'type16'
    ]
    
    results = {}
    
    for psekraac_type in psekraac_types:
        print(f"\n  Extracting PseKRAAC_{psekraac_type}...", end=" ", flush=True)
        
        output_file = os.path.join(output_dir, f"PseKRAAC_{psekraac_type}.tsv")
        
        try:
            cmd = [
                "python",
                str(IFeaturePseKRAAC_SCRIPT),
                "--file", input_file,
                "--type", psekraac_type,
                "--subtype", "g-gap",
                "--ktuple", "2",
                "--gap_lambda", "2",
                "--raactype", "5",
                "--out", output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and os.path.exists(output_file):
                df = pd.read_csv(output_file, sep='\t')
                results[psekraac_type] = df
                print(f"OK ({df.shape[1] - 1} features)")
            else:
                print("FAIL")
                
        except subprocess.TimeoutExpired:
            print("FAIL: Timeout")
        except Exception as e:
            print(f"FAIL: {e}")
    
    print(f"\n  PseKRAAC extraction completed: {len(results)}/{len(psekraac_types)} successful")
    return results


# ============================================================================
# modlAMP特征提取
# ============================================================================
def extract_modlamp_features(input_file, output_dir):
    """使用modlAMP提取肽描述符特征"""
    print("\n" + "=" * 70)
    print("Step 3: Extracting modlAMP Features")
    print("=" * 70)
    
    from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
    
    os.makedirs(output_dir, exist_ok=True)
    
    ids, sequences = parse_fasta(input_file)
    print(f"  Loaded {len(sequences)} sequences")
    
    results = {}
    
    # 1. 全局描述符
    print("\n  Extracting Global Descriptors...", end=" ", flush=True)
    try:
        glob = GlobalDescriptor(sequences)
        glob.isoelectric_point()
        isoelectric_point = glob.descriptor.flatten()
        
        glob.length()
        length = glob.descriptor.flatten()
        
        glob.hydrophobic_ratio()
        hydrophobic_ratio = glob.descriptor.flatten()
        
        global_df = pd.DataFrame({
            'ID': ids,
            'modlamp_Length': length,
            'modlamp_Isoelectric_Point': isoelectric_point,
            'modlamp_Hydrophobic_Ratio': hydrophobic_ratio
        })
        
        global_file = os.path.join(output_dir, "modlamp_global.tsv")
        global_df.to_csv(global_file, sep='\t', index=False)
        
        results['global'] = global_df
        print(f"OK (3 features)")
    except Exception as e:
        print(f"FAIL: {e}")
    
    # 2. Eisenberg疏水矩
    print("  Extracting Eisenberg Moment...", end=" ", flush=True)
    try:
        desc = PeptideDescriptor(sequences, 'eisenberg')
        desc.calculate_moment()
        
        eisenberg_df = pd.DataFrame({
            'ID': ids,
            'modlamp_Eisenberg_Moment': desc.descriptor.flatten()
        })
        
        eisenberg_file = os.path.join(output_dir, "modlamp_eisenberg_moment.tsv")
        eisenberg_df.to_csv(eisenberg_file, sep='\t', index=False)
        
        results['eisenberg_moment'] = eisenberg_df
        print(f"OK (1 feature)")
    except Exception as e:
        print(f"FAIL: {e}")
    
    # 3. Kyte-Doolittle疏水矩
    print("  Extracting Kyte-Doolittle Moment...", end=" ", flush=True)
    try:
        desc = PeptideDescriptor(sequences, 'kytedoolittle')
        desc.calculate_moment()
        
        kd_df = pd.DataFrame({
            'ID': ids,
            'modlamp_KyteDoolittle_Moment': desc.descriptor.flatten()
        })
        
        kd_file = os.path.join(output_dir, "modlamp_kd_moment.tsv")
        kd_df.to_csv(kd_file, sep='\t', index=False)
        
        results['kd_moment'] = kd_df
        print(f"OK (1 feature)")
    except Exception as e:
        print(f"FAIL: {e}")
    
    # 4. 自相关特征
    print("  Extracting Autocorrelation...", end=" ", flush=True)
    try:
        desc = PeptideDescriptor(sequences, 'eisenberg')
        desc.calculate_autocorr(7)
        
        autocorr_df = pd.DataFrame(
            desc.descriptor,
            columns=[f'modlamp_AutoCorr_{i}' for i in range(7)]
        )
        autocorr_df.insert(0, 'ID', ids)
        
        autocorr_file = os.path.join(output_dir, "modlamp_autocorrelation.tsv")
        autocorr_df.to_csv(autocorr_file, sep='\t', index=False)
        
        results['autocorrelation'] = autocorr_df
        print(f"OK (7 features)")
    except Exception as e:
        print(f"FAIL: {e}")
    
    # 5. ARC特征 (Arc/Coil/Helix结构)
    print("  Extracting ARC (Arc/Coil/Helix)...", end=" ", flush=True)
    try:
        desc = PeptideDescriptor(sequences, 'eisenberg')
        desc.calculate_arc()
        
        arc_df = pd.DataFrame(
            desc.descriptor,
            columns=['modlamp_ARC_Helix', 'modlamp_ARC_Coil', 'modlamp_ARC_Other']
        )
        arc_df.insert(0, 'ID', ids)
        
        arc_file = os.path.join(output_dir, "modlamp_arc.tsv")
        arc_df.to_csv(arc_file, sep='\t', index=False)
        
        results['arc'] = arc_df
        print(f"OK (3 features)")
    except Exception as e:
        print(f"FAIL: {e}")
    
    # 6. 互相关特征
    print("  Extracting Cross-Correlation...", end=" ", flush=True)
    try:
        desc = PeptideDescriptor(sequences, 'eisenberg')
        desc.calculate_crosscorr(7)
        
        crosscorr_df = pd.DataFrame(
            desc.descriptor,
            columns=[f'modlamp_CrossCorr_{i}' for i in range(49)]
        )
        crosscorr_df.insert(0, 'ID', ids)
        
        crosscorr_file = os.path.join(output_dir, "modlamp_crosscorr.tsv")
        crosscorr_df.to_csv(crosscorr_file, sep='\t', index=False)
        
        results['crosscorr'] = crosscorr_df
        print(f"OK (49 features)")
    except Exception as e:
        print(f"FAIL: {e}")
    
    print(f"\n  modlAMP extraction completed: {len(results)} types successful")
    return results


# ============================================================================
# 合并所有特征
# ============================================================================
def combine_all_features(ifeature_results, psekraac_results, modlamp_results, ids, output_file):
    """将所有特征合并为一个CSV文件"""
    print("\n" + "=" * 70)
    print("Step 4: Combining All Features")
    print("=" * 70)
    
    # 首先创建包含ID的基础DataFrame
    combined_df = pd.DataFrame({'ID': ids})
    total_features = 0
    
    # 处理iFeature特征
    for feature_name, df in ifeature_results.items():
        id_col = df.columns[0]
        feature_cols = [col for col in df.columns if col != id_col]
        new_cols = {col: f'iFeature_{feature_name}_{col}' for col in feature_cols}
        feature_df = df.rename(columns=new_cols)
        feature_df = feature_df[list(new_cols.values())]
        combined_df = pd.concat([combined_df, feature_df], axis=1)
        total_features += len(feature_cols)
        print(f"  Added iFeature_{feature_name}: {len(feature_cols)} features")
    
    # 处理PseKRAAC特征
    for feature_name, df in psekraac_results.items():
        id_col = df.columns[0]
        feature_cols = [col for col in df.columns if col != id_col]
        new_cols = {col: f'PseKRAAC_{feature_name}_{col}' for col in feature_cols}
        feature_df = df.rename(columns=new_cols)
        feature_df = feature_df[list(new_cols.values())]
        combined_df = pd.concat([combined_df, feature_df], axis=1)
        total_features += len(feature_cols)
        print(f"  Added PseKRAAC_{feature_name}: {len(feature_cols)} features")
    
    # 处理modlAMP特征
    for feature_name, df in modlamp_results.items():
        if 'ID' in df.columns:
            feature_cols = [col for col in df.columns if col != 'ID']
            new_cols = {col: f'{col}' for col in feature_cols}
            feature_df = df.rename(columns=new_cols)
            feature_df = feature_df[list(new_cols.values())]
            combined_df = pd.concat([combined_df, feature_df], axis=1)
            total_features += len(feature_cols)
            print(f"  Added {feature_name}: {len(feature_cols)} features")
    
    # 保存为CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n  Total sequences: {len(combined_df)}")
    print(f"  Total features: {total_features}")
    print(f"  Output file: {output_file}")
    print(f"  Combined dataframe shape: {combined_df.shape}")
    
    return combined_df


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    print("=" * 70)
    print("Protein Feature Extraction Script")
    print("For sequences_with_labels.csv")
    print("=" * 70)
    print(f"\nInput CSV: {CSV_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Step 0: 创建合并的FASTA文件
    input_file = create_combined_fasta()
    
    # 解析输入文件获取序列ID
    ids, sequences = parse_fasta(input_file)
    print(f"\nLoaded {len(sequences)} sequences from FASTA file")
    
    # Step 1: 提取iFeature特征
    ifeature_output = str(OUTPUT_DIR / "ifeature")
    ifeature_results = extract_ifeature_features(input_file, ifeature_output)
    
    # Step 2: 提取PseKRAAC特征
    psekraac_output = str(OUTPUT_DIR / "psekraac")
    psekraac_results = extract_psekraac_features(input_file, psekraac_output)
    
    # Step 3: 提取modlAMP特征
    modlamp_output = str(OUTPUT_DIR / "modlamp")
    modlamp_results = extract_modlamp_features(input_file, modlamp_output)
    
    # Step 4: 合并所有特征
    combined_df = combine_all_features(
        ifeature_results, 
        psekraac_results, 
        modlamp_results, 
        ids, 
        OUTPUT_CSV
    )
    
    # 清理临时FASTA文件
    if TEMP_FASTA.exists():
        TEMP_FASTA.unlink()
        print(f"\n  Cleaned up temporary FASTA file")
    
    print("\n" + "=" * 70)
    print("Feature Extraction Completed Successfully!")
    print("=" * 70)
    
    return combined_df


if __name__ == "__main__":
    main()
