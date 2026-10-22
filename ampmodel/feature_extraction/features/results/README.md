# 特征提取结果目录

## 目录结构

本目录包含从氨基酸序列中提取的各种特征，用于后续的机器学习模型训练。目录结构如下：

```
results/
├── ifeature/              # iFeature工具生成的特征文件
├── modlamp/               # modlamp工具生成的特征文件
├── psekraac/              # PseKRAAC工具生成的特征文件
├── all_features_combined.csv      # 所有特征的组合
├── feature_names.txt              # 特征名称列表
├── features_processed.csv         # 处理后的特征文件
├── features_standardized.csv      # 标准化后的特征
├── features_variance_filtered.csv # 经过方差过滤的特征
├── input_feature_names.txt        # 输入特征名称
├── labels.csv                     # 标签文件
└── sequences_with_labels.csv      # 带标签的序列文件
```

## 文件说明

### 特征文件

#### iFeature 特征
`ifeature/` 目录包含使用iFeature工具提取的各种特征：
- `iFeature_AAC.tsv` - 氨基酸组成特征
- `iFeature_APAAC.tsv` - 自相关伪氨基酸组成特征
- `iFeature_CKSAAGP.tsv` - 带间隙的k-spaced氨基酸对组成特征
- `iFeature_CKSAAP.tsv` - k-spaced氨基酸对组成特征
- `iFeature_CTDC.tsv` - 组成、转换、分布特征（组成部分）
- `iFeature_CTDD.tsv` - 组成、转换、分布特征（分布部分）
- `iFeature_CTDT.tsv` - 组成、转换、分布特征（转换部分）
- `iFeature_CTriad.tsv` - 三肽组成特征
- `iFeature_DDE.tsv` - 二肽组成特征
- `iFeature_DPC.tsv` - 二肽组成特征
- `iFeature_GAAC.tsv` - 分组氨基酸组成特征
- `iFeature_GDPC.tsv` - 分组二肽组成特征
- `iFeature_GTPC.tsv` - 分组三肽组成特征
- `iFeature_Geary.tsv` - Geary自相关特征
- `iFeature_KSCTriad.tsv` - k-spaced三肽组成特征
- `iFeature_Moran.tsv` - Moran自相关特征
- `iFeature_NMBroto.tsv` - NMBroto自相关特征
- `iFeature_PAAC.tsv` - 伪氨基酸组成特征

#### modlamp 特征
`modlamp/` 目录包含使用modlamp工具提取的特征：
- `modlamp_autocorrelation.tsv` - 自相关特征
- `modlamp_eisenberg_moment.tsv` - Eisenberg矩特征
- `modlamp_global.tsv` - 全局特征（长度、等电点、疏水比例等）
- `modlamp_kd_moment.tsv` - KD矩特征

#### PseKRAAC 特征
`psekraac/` 目录包含使用PseKRAAC工具提取的各种类型的特征：
- `PseKRAAC_type1.tsv` 至 `PseKRAAC_type16.tsv` - 不同类型的伪氨基酸组成特征

### 汇总文件

- `all_features_combined.csv` - 所有特征的组合，包含从各个工具提取的特征（特征数量：4921）
- `feature_names.txt` - 所有特征的名称列表
- `features_processed.csv` - 处理后的特征文件，包含标准化和过滤后的特征（特征数量：1098）
- `features_standardized.csv` - 标准化后的特征文件（特征数量：1489）
- `features_variance_filtered.csv` - 经过方差过滤的特征文件，去除了低方差的特征（特征数量：1489）
- `feature_original_stats.csv` - 标准化前特征的均值和标准差，用于逆向还原标准化过程（特征数量：1489）
- `input_feature_names.txt` - 输入特征的名称列表
- `labels.csv` - 样本的标签文件
- `sequences_with_labels.csv` - 带标签的序列文件，包含原始序列和对应的标签

## 特征说明

本目录中的特征文件包含以下类型的特征：

1. **氨基酸组成特征** - 描述蛋白质序列中各种氨基酸的出现频率
2. **伪氨基酸组成特征** - 考虑了氨基酸的物理化学性质和序列顺序信息
3. **二肽和三肽组成特征** - 描述氨基酸对和三肽的出现频率
4. **组成、转换、分布特征** - 描述氨基酸的组成、转换和分布情况
5. **自相关特征** - 描述氨基酸物理化学性质的自相关情况
6. **全局特征** - 如序列长度、等电点、疏水比例等

这些特征可以用于训练机器学习模型，预测蛋白质的功能、结构或其他特性。

## 使用方法

1. 可以直接使用 `features_processed.csv` 文件作为模型的输入特征
2. 如需使用特定类型的特征，可以从对应的子目录中获取原始特征文件
3. `feature_names.txt` 文件包含所有特征的名称，可用于特征选择和分析

## 注意事项

- 特征文件中的特征值已经过标准化处理，可直接用于模型训练
- 特征文件中的第一列通常是样本ID，用于与标签文件对应
- 标签文件中的标签值通常是二进制的（0或1），表示样本的类别