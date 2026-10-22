import pandas as pd
from collections import defaultdict, Counter


def analyze_mmseq_clustering():
    cluster_file = "mmseq/non_antimicrobial_peptides_high_quality_complete_cluster.tsv"
    
    print("=" * 60)
    print("MMseq2聚类结果详细分析")
    print("=" * 60)
    
    print(f"\n正在加载聚类文件: {cluster_file}")
    cluster_df = pd.read_csv(cluster_file, sep='\t', header=None, names=['rep_seq_id', 'rep_seq_name', 'target_seq_id', 'target_seq_name'])
    print(f"聚类文件加载完成，共 {len(cluster_df)} 条记录")
    
    print("\n" + "=" * 60)
    print("聚类关系分析")
    print("=" * 60)
    
    cluster_mapping = defaultdict(list)
    for _, rep_name, target_id, _ in cluster_df.values:
        cluster_mapping[rep_name].append(target_id)
    
    unique_clusters = len(cluster_mapping)
    print(f"\n1. 聚类关系统计:")
    print(f"   总记录数: {len(cluster_df)}")
    print(f"   唯一聚类数: {unique_clusters}")
    print(f"   平均每个聚类的序列数: {len(cluster_df)/unique_clusters:.1f}")
    
    cluster_sizes = {k: len(set(v)) for k, v in cluster_mapping.items()}
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n2. 聚类大小分布:")
    size_distribution = Counter(cluster_sizes.values())
    for size, count in sorted(size_distribution.items(), key=lambda x: x[0]):
        percentage = (count / unique_clusters) * 100
        print(f"   大小 {size}: {count} 个聚类 ({percentage:.1f}%)")
    
    print(f"\n3. 大型聚类 (大小>=2):")
    large_clusters = [(name, size) for name, size in cluster_sizes.items() if size >= 2]
    large_clusters_sorted = sorted(large_clusters, key=lambda x: x[1], reverse=True)
    print(f"   大型聚类数: {len(large_clusters)}")
    for i, (cluster_name, size) in enumerate(large_clusters_sorted[:10], 1):
        percentage = (size / len(cluster_df)) * 100
        print(f"   {i}. {cluster_name}: {size} 条序列 ({percentage:.1f}%)")
    
    print(f"\n4. 单序列聚类 (大小=1):")
    single_clusters = [name for name, size in cluster_sizes.items() if size == 1]
    print(f"   单序列聚类数: {len(single_clusters)}")
    print(f"   占比: {len(single_clusters)/unique_clusters*100:.1f}%")
    
    print(f"\n5. 聚类名称分析 (Top 20):")
    cluster_name_counter = Counter([row[1] for row in cluster_df.values])
    for name, count in cluster_name_counter.most_common(20):
        percentage = (count / len(cluster_df)) * 100
        print(f"   {name}: {count} 条记录 ({percentage:.1f}%)")
    
    print(f"\n6. 聚类代表性分析:")
    if len(large_clusters) > 0:
        print(f"   发现 {len(large_clusters)} 个包含多个序列的聚类")
        print(f"   这些聚类可能代表同一家族或功能相似的序列")
        total_in_large = sum([size for _, size in large_clusters])
        print(f"   大型聚类包含序列总数: {total_in_large}")
        print(f"   占总序列比例: {total_in_large/len(cluster_df)*100:.1f}%")
    else:
        print(f"   未发现包含多个序列的聚类")
        print(f"   所有序列都是独立的，没有聚类效果")
    
    print(f"\n7. 序列相似度分析:")
    print(f"   如果序列高度相似，应该形成聚类")
    print(f"   如果所有聚类大小都为1，说明序列多样性很高")
    
    print("\n" + "=" * 60)
    print("聚类分析完成！")
    print("=" * 60)
    
    print(f"\n输出文件:")
    print(f"  {cluster_file} - 聚类结果TSV文件")
    
    print(f"\n结论:")
    if len(large_clusters) == 0:
        print(f"  - 所有4281条序列都是独立的，没有聚类")
        print(f"  - 序列多样性很高，每条序列都是独特的")
        print(f"  - 可能原因：")
        print(f"    1. 序列长度差异大（2-60 aa）")
        print(f"    2. 序列来自不同物种和功能")
        print(f"    3. MMseq2参数设置过于严格")
    else:
        print(f"  - 发现 {len(large_clusters)} 个聚类")
        print(f"  - 可以使用代表序列减少数据冗余")
        print(f"  - 建议进一步分析大型聚类的功能特征")


if __name__ == "__main__":
    analyze_mmseq_clustering()
