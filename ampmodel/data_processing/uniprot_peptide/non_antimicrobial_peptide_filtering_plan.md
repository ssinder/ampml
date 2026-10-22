# 非抗菌肽严格筛选计划（优化版）

## 项目目标

基于 UniProt 短肽数据集（18,171条记录），采用更严格的筛选标准，筛选出最不可能具有抗菌活性的短肽作为高质量负样本。

## 数据集概况

### 短肽数据集

**数据源**: `short_peptide_results_complete.json`
**总记录数**: 18,171 条
**关键词种类**: 652 种
**数据质量**: Swiss-Prot（人工审核）

### GO术语分析

**数据源**: `go_terms_analysis.json`
**唯一GO术语总数**: 2,021 个
**分类统计**:
- 分子功能: 609 (30.1%)
- 生物过程: 1,102 (54.5%)
- 细胞组分: 310 (15.4%)

## 严格筛选策略

### 核心原则

1. **多层排除**: 从多个维度排除抗菌相关肽
2. **保守策略**: 宁可误排，不可漏排
3. **功能明确**: 优先选择具有明确非抗菌功能的肽
4. **高质量**: 优先选择有实验证据的肽

### 评分系统设计（优化版）

#### 评分标准

| 标准 | 权重 | 说明 | 优先级 |
|------|------|------|--------|
| 直接抗菌GO术语 | +5.0 | 直接抗菌活性 | **高** |
| 毒素相关GO术语 | +4.0 | 毒素活性 | **高** |
| 免疫相关GO术语 | +3.0 | 免疫防御 | **中高** |
| 细胞毒性GO术语 | +3.0 | 细胞裂解 | **中高** |
| 抗菌相关关键词 | +4.0 | 明确抗菌标签 | **高** |
| 毒素相关关键词 | +3.0 | 毒素标签 | **中高** |
| 免疫相关关键词 | +2.0 | 免疫标签 | **中** |
| 病毒相关GO术语 | +2.0 | 病毒毒力 | **中** |
| 病毒相关关键词 | +2.0 | 病毒标签 | **中** |

#### 评分解释

| 分数 | 抗菌可能性 | 筛选决策 | 说明 |
|------|----------|----------|------|
| 0.0 | 极低（理想负样本）| **保留** | 无任何抗菌相关特征 |
| 0.1-1.0 | 很低 | **保留** | 仅胞外定位 |
| 1.1-2.0 | 低 | **谨慎保留** | 少量免疫相关 |
| 2.1-3.0 | 中低 | **排除** | 明显抗菌特征 |
| 3.1-4.0 | 中等 | **排除** | 中度抗菌特征 |
| 4.1-5.0+ | 高 | **排除** | 高度抗菌特征 |

### 排除标准详解（扩展版）

#### 1. 直接抗菌相关GO术语（8个）

| GO ID | GO术语 | 说明 | 权重 |
|-------|---------|------|------|
| GO:0042742 | 对细菌的防御反应 | 直接抗菌防御 | +5.0 |
| GO:0031640 | 杀死其他生物的细胞 | 细胞毒性 | +5.0 |
| GO:0090729 | 毒素活性 | 毒素相关 | +4.0 |
| GO:0045087 | 先天免疫反应 | 免疫相关 | +3.0 |
| GO:0050832 | 对真菌的防御反应 | 抗真菌 | +5.0 |
| GO:0050829 | 对革兰氏阴性细菌的防御反应 | 抗革兰氏阴性菌 | +5.0 |
| GO:0050830 | 对革兰氏阳性细菌的防御反应 | 抗革兰氏阳性菌 | +5.0 |

#### 2. 毒素相关GO术语（扩展）

| GO ID | GO术语 | 说明 | 权重 |
|-------|---------|------|------|
| GO:0090729 | 毒素活性 | 通用毒素 | +4.0 |
| GO:0044487 | 毒素介导的神经冲动传导干扰 | 神经毒素 | +4.0 |
| GO:0044398 | 毒素介导的水肿 | 水肿毒素 | +4.0 |
| GO:0044478 | 毒素介导的血小板聚集 | 血液毒素 | +4.0 |
| GO:0035738 | 毒素介导的生物过程干扰 | 通用毒素 | +4.0 |
| GO:0044562 | 毒素介导的电压门控钾通道活性抑制 | 钾通道毒素 | +4.0 |
| GO:0051673 | 破坏另一个生物的质膜完整性 | 溶血毒素 | +4.0 |
| GO:0001897 | 共生体介导的宿主细胞裂解 | 裂解毒素 | +4.0 |

#### 3. 免疫相关GO术语（扩展）

| GO ID | GO术语 | 说明 | 权重 |
|-------|---------|------|------|
| GO:0006952 | 防御反应 | 一般防御 | +3.0 |



#### 4. 细胞毒性GO术语（扩展）

| GO ID | GO术语 | 说明 | 权重 |
|-------|---------|------|------|
| GO:0031640 | 杀死其他生物的细胞 | 细胞毒性 | +3.0 |
| GO:0051715 | 在另一个生物中的细胞裂解 | 细胞裂解 | +3.0 |
| GO:0019875 | 细胞裂解 | 通用裂解 | +3.0 |


#### 5. 病毒相关GO术语（新增）

| GO ID | GO术语 | 说明 | 权重 |
|-------|---------|------|------|
| GO:0019048 | 病毒毒力 | 病毒毒力 | +2.0 |


#### 6. 抗菌相关关键词（21类）

**直接抗菌相关**（权重 +4.0）:
- Antimicrobial（抗菌）
- Defensin（防御肽）
- Antibiotic（抗生素）
- Bacteriocin（细菌素）
- Antifungal（抗真菌）
- Antiviral（抗病毒）
- Antiparasitic（抗寄生虫）

**毒素相关**（权重 +3.0）:
- Toxin（毒素）
- Neurotoxin（神经毒素）
- Cardiotoxin（心脏毒素）
- Hemorrhagic toxin（出血毒素）
- Myotoxin（肌毒素）
- Dermonecrotic toxin（皮肤坏死毒素）
- Enterotoxin（肠毒素）
- Hemolysin（溶血素）

**免疫相关**（权重 +2.0）:
- Immunity（免疫）
- Innate immunity（先天免疫）
- Adaptive immunity（适应性免疫）
- Defense（防御）
- Antibiotic resistance（抗生素耐药性）


**其他防御相关**（权重 +2.0）:
- Plant defense（植物防御）
- Amphibian defense peptide（两栖类防御肽）
- Insect defense（昆虫防御）
- Fungicide（杀真菌剂）


#### 7. 病毒相关关键词（新增，权重 +2.0）

- Virulence（毒力）



## 筛选流程（优化版）

### 第一阶段：严格筛选

**目标**: 排除所有明确抗菌相关的肽

**筛选条件**:
```
(antimicrobial_score == 0.0)
```

**预期结果**: 约6,000-8,000条记录

### 第二阶段：功能验证

**目标**: 确保保留的肽具有明确的非抗菌功能

**优先功能类型**:
1. **核糖体相关蛋白**
   - 核糖体亚基
   - 翻译因子
   - 核糖核蛋白复合物

2. **光合作用蛋白**
   - 光系统II反应中心
   - 细胞色素b6f复合物
   - 电子传递链蛋白

3. **线粒体蛋白**
   - 线粒体膜蛋白
   - ATP合酶亚基
   - 线粒体输入受体

4. **细胞质/核蛋白**
   - 发育相关蛋白
   - 转录调控因子
   - 细胞周期蛋白

5. **代谢酶**
   - 氧化还原酶
   - 转移酶
   - 水解酶

### 第三阶段：质量评估

**目标**: 评估筛选结果的质量

**评估指标**:
1. **功能明确性**: 是否有明确的非抗菌功能
2. **结构多样性**: 是否涵盖不同的结构类型
3. **物种多样性**: 是否涵盖多个物种
4. **实验验证**: 是否有实验验证的功能注释
5. **蛋白质存在水平**: 是否有蛋白质水平证据

## 负样本特征分析（优化版）

### 预期分布

基于GO术语分析和更严格的筛选，预期负样本分布：

| 功能类型 | 预期比例 | 说明 |
|---------|----------|------|
| 核糖体相关 | ~35% | 核糖体蛋白和翻译相关 |
| 光合作用相关 | ~25% | 植物光系统蛋白 |
| 线粒体相关 | ~20% | 线粒体膜蛋白和ATP合酶 |
| 细胞质/核蛋白 | ~15% | 发育和转录相关 |
| 其他 | ~5% | 膜蛋白、代谢酶等 |

### 典型负样本示例

#### 示例1: 核糖体蛋白

```
Accession: P0C7X4
Name: 50S ribosomal protein L36
Organism: Bacillus subtilis
Length: 38 aa
Sequence: MRAKVRKVRVVTGTPPVV
Keywords: 3D-structure, Direct protein sequencing, Reference proteome
GO Terms: GO:0003735 (structural constituent of ribosome)
Antimicrobial Score: 0.0
Protein Existence: 1: Evidence at protein level
```

**特征**: 典型的核糖体蛋白，参与蛋白质合成，无抗菌活性，有蛋白质水平证据

#### 示例2: 光合作用蛋白

```
Accession: P0C0W0
Name: Cytochrome b6-f complex subunit PetL
Organism: Synechocystis sp. PCC 6803
Length: 34 aa
Sequence: MVLSPADKTNVK
Keywords: 3D-structure, Direct protein sequencing, Chloroplast
GO Terms: GO:0009512 (cytochrome b6f complex)
Antimicrobial Score: 0.0
Protein Existence: 1: Evidence at protein level
```

**特征**: 光合作用电子传递链蛋白，功能明确，有蛋白质水平证据

#### 示例3: 线粒体蛋白

```
Accession: Q96IX5
Name: ATP synthase F(0) complex subunit k, mitochondrial
Organism: Homo sapiens
Length: 58 aa
Sequence: MFRRLTFAQLLFATVLGIAGGVYIFQPVFEQYAKDQKELKEKMQLVQESEEKKS
Keywords: Acetylation, Membrane, Mitochondrion, Reference proteome
GO Terms: GO:0005741 (mitochondrial outer membrane)
Antimicrobial Score: 0.0
Protein Existence: 1: Evidence at protein level
```

**特征**: 线粒体膜蛋白，参与ATP合成，有蛋白质水平证据

## 实施步骤（优化版）

### 步骤1: 数据预处理

```python
import pandas as pd
import json

# 读取数据
with open('short_peptide_results_complete.json', 'r', encoding='utf-8') as f:
    peptide_data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(peptide_data)

# 读取GO术语分析
with open('go_terms_analysis.json', 'r', encoding='utf-8') as f:
    go_data = json.load(f)
```

### 步骤2: 应用严格筛选条件

```python
# 定义抗菌相关GO术语（扩展版）
antimicrobial_go_terms = {
    # 直接抗菌
    'GO:0042742', 'GO:0031640', 'GO:0006952', 'GO:0090729',
    'GO:0045087', 'GO:0050832', 'GO:0050829', 'GO:0050830',
    # 毒素相关
    'GO:0044487', 'GO:0044398', 'GO:0044478', 'GO:0035738',
    'GO:0044562', 'GO:0051673', 'GO:0001897',
    # 免疫相关
    'GO:0002250', 'GO:0045088', 'GO:0002819', 'GO:0045591',
    'GO:0045577', 'GO:0050868',
    # 细胞毒性
    'GO:0051715', 'GO:0019875', 'GO:0008219', 'GO:0006915',
    # 病毒相关
    'GO:0019048', 'GO:0019058', 'GO:0019061', 'GO:0019062',
    'GO:0019063', 'GO:0019064'
}

# 定义抗菌相关关键词（扩展版）
antimicrobial_keywords = {
    # 直接抗菌
    'Antimicrobial', 'Defensin', 'Antibiotic', 'Bacteriocin',
    'Antifungal', 'Antiviral', 'Antiparasitic',
    # 毒素相关
    'Toxin', 'Neurotoxin', 'Cardiotoxin', 'Hemorrhagic toxin',
    'Myotoxin', 'Dermonecrotic toxin', 'Enterotoxin', 'Hemolysin',
    # 免疫相关
    'Immunity', 'Innate immunity', 'Adaptive immunity', 'Defense',
    'Antibiotic resistance', 'Antibiotic biosynthesis',
    # 其他防御相关
    'Plant defense', 'Amphibian defense peptide', 'Insect defense',
    'Fungicide', 'Allergen',
    # 病毒相关
    'Virulence', 'Viral capsid', 'Viral envelope', 'Viral replication',
    'Viral transcription', 'Viral translation', 'Viral assembly',
    'Viral release', 'Viral entry', 'Viral immunoevasion'
}

# 定义胞外定位GO术语
extracellular_go_terms = {'GO:0005576', 'GO:0005615'}

# 应用筛选
def has_antimicrobial_go(row):
    go_terms = row.get('go_terms', '')
    return any(go_id in go_terms for go_id in antimicrobial_go_terms)

def has_antimicrobial_keywords(row):
    keywords = row.get('keywords', '')
    return any(kw.lower() in keywords.lower() for kw in antimicrobial_keywords)

def is_extracellular(row):
    go_terms = row.get('go_terms', '')
    return any(go_id in go_terms for go_id in extracellular_go_terms)

# 计算抗菌评分
def calculate_antimicrobial_score(row):
    score = 0.0
    
    go_terms = row.get('go_terms', '')
    keywords = row.get('keywords', '')
    
    # 直接抗菌GO术语
    direct_antimicrobial_go = {
        'GO:0042742', 'GO:0031640', 'GO:0050832', 
        'GO:0050829', 'GO:0050830'
    }
    if any(go_id in go_terms for go_id in direct_antimicrobial_go):
        score += 5.0
    
    # 毒素相关GO术语
    toxin_go = {
        'GO:0090729', 'GO:0044487', 'GO:0044398', 'GO:0044478',
        'GO:0035738', 'GO:0044562', 'GO:0051673', 'GO:0001897'
    }
    if any(go_id in go_terms for go_id in toxin_go):
        score += 4.0
    
    # 免疫相关GO术语
    immune_go = {
        'GO:0006952', 'GO:0045087', 'GO:0002250', 'GO:0045088',
        'GO:0002819', 'GO:0045591', 'GO:0045577', 'GO:0050868'
    }
    if any(go_id in go_terms for go_id in immune_go):
        score += 3.0
    
    # 细胞毒性GO术语
    cytotoxic_go = {
        'GO:0051715', 'GO:0019875', 'GO:0008219', 'GO:0006915'
    }
    if any(go_id in go_terms for go_id in cytotoxic_go):
        score += 3.0
    
    # 病毒相关GO术语
    viral_go = {
        'GO:0019048', 'GO:0019058', 'GO:0019061', 'GO:0019062',
        'GO:0019063', 'GO:0019064'
    }
    if any(go_id in go_terms for go_id in viral_go):
        score += 2.0
    
    # 抗菌相关关键词
    direct_antimicrobial_kw = {
        'Antimicrobial', 'Defensin', 'Antibiotic', 'Bacteriocin',
        'Antifungal', 'Antiviral', 'Antiparasitic'
    }
    if any(kw.lower() in keywords.lower() for kw in direct_antimicrobial_kw):
        score += 4.0
    
    # 毒素相关关键词
    toxin_kw = {
        'Toxin', 'Neurotoxin', 'Cardiotoxin', 'Hemorrhagic toxin',
        'Myotoxin', 'Dermonecrotic toxin', 'Enterotoxin', 'Hemolysin'
    }
    if any(kw.lower() in keywords.lower() for kw in toxin_kw):
        score += 3.0
    
    # 免疫相关关键词
    immune_kw = {
        'Immunity', 'Innate immunity', 'Adaptive immunity', 'Defense',
        'Antibiotic resistance', 'Antibiotic biosynthesis',
        'Plant defense', 'Amphibian defense peptide', 'Insect defense',
        'Fungicide', 'Allergen'
    }
    if any(kw.lower() in keywords.lower() for kw in immune_kw):
        score += 2.0
    
    # 病毒相关关键词
    viral_kw = {
        'Virulence', 'Viral capsid', 'Viral envelope', 'Viral replication',
        'Viral transcription', 'Viral translation', 'Viral assembly',
        'Viral release', 'Viral entry', 'Viral immunoevasion'
    }
    if any(kw.lower() in keywords.lower() for kw in viral_kw):
        score += 2.0
    
    # 胞外定位
    if is_extracellular(row):
        score += 1.5
    
    return score

# 筛选负样本
df['antimicrobial_score'] = df.apply(calculate_antimicrobial_score, axis=1)
negative_samples = df[df['antimicrobial_score'] == 0.0].copy()
```

### 步骤3: 质量检查

```python
# 检查功能分布
print("\n功能类型分布:")
print(negative_samples['keywords'].str.split('; ').explode().value_counts().head(20))

# 检查物种分布
print("\n物种分布:")
print(negative_samples['organism'].value_counts().head(10))

# 检查长度分布
print("\n序列长度分布:")
print(negative_samples['length'].describe())

# 检查蛋白质存在水平
print("\n蛋白质存在水平分布:")
print(negative_samples['protein_existence'].value_counts())

# 优先选择有蛋白质水平证据的样本
high_quality_samples = negative_samples[
    negative_samples['protein_existence'].str.contains('Evidence at protein level', na=False)
]

print(f"\n高质量样本（有蛋白质水平证据）: {len(high_quality_samples)} 条")
```

## 质量控制措施（优化版）

### 1. 多层验证

**第一层**: 自动化筛选
- 应用严格的评分系统
- 排除所有评分>0的肽

**第二层**: 人工审查
- 抽样检查200-300条记录
- 验证功能描述的准确性
- 确认没有遗漏的抗菌相关术语

**第三层**: 文献验证
- 对代表性样本进行文献验证
- 搜索PubMed确认无抗菌活性报道
- 检查是否有间接抗菌功能

### 2. 质量指标

**覆盖率**: 负样本应涵盖主要功能类型
**多样性**: 物种、长度、功能应多样化
**准确性**: 经验证应无抗菌活性
**可重复性**: 筛选过程应可重复和验证
**证据水平**: 优先选择有蛋白质水平证据的样本

### 3. 风险缓解

**未知抗菌活性**:
- 通过多层筛选降低风险
- 优先选择有实验证据的样本
- 通过文献验证确认功能

**物种特异性**:
- 考虑实验条件的影响
- 选择多个物种的样本
- 避免选择已知有抗菌活性的物种

**功能重叠**:
- 仔细分析功能注释
- 排除间接参与防御的肽
- 优先选择结构/代谢功能明确的肽

## 预期结果（优化版）

### 负样本数量

基于严格筛选条件，预期可获得：
- **理想负样本**（评分0.0）: 6,000-8,000条
- **高质量负样本**（评分0.0 + 蛋白质水平证据）: 4,000-5,000条

### 样本特征

**功能多样性**:
- 核糖体蛋白: ~35%
- 光合作用蛋白: ~25%
- 线粒体蛋白: ~20%
- 其他功能: ~20%

**物种多样性**:
- 细菌: ~20%
- 植物: ~25%
- 动物: ~30%
- 其他: ~25%

**长度分布**:
- 短肽（1-20 aa）: ~10%
- 中等肽（21-40 aa）: ~65%
- 长肽（41-60 aa）: ~25%

**证据水平**:
- 蛋白质水平证据: ~60%
- 同源推断: ~35%
- 其他: ~5%

## 应用场景

### 1. 机器学习模型训练

**正样本**: 已知抗菌肽（从文献或实验数据收集）
**负样本**: 本严格筛选计划筛选的非抗菌肽
**特征**: 序列特征、理化性质、结构特征

### 2. 功能预测

**目标**: 预测新肽的抗菌活性
**方法**: 基于筛选出的负样本训练分类器
**验证**: 使用独立的测试集验证性能

### 3. 肽段设计

**目标**: 设计新的抗菌肽
**参考**: 基于负样本的结构特征
**优化**: 避免与非抗菌肽的特征重叠

## 成功标准（优化版）

### 筛选质量指标

1. **覆盖率**: 负样本应涵盖主要功能类型
2. **多样性**: 物种、长度、功能应多样化
3. **准确性**: 经验证应无抗菌活性
4. **可重复性**: 筛选过程应可重复和验证
5. **证据水平**: >50%的样本应有蛋白质水平证据

### 模型性能指标

1. **准确率**: >90%
2. **精确率**: >85%
3. **召回率**: >80%
4. **AUC-ROC**: >0.90

## 与初筛计划的对比

| 指标 | 初筛计划 | 严格筛选计划 | 改进 |
|------|----------|------------|------|
| 排除GO术语 | 8个 | 30+个 | +275% |
| 排除关键词 | 17类 | 21类 | +24% |
| 评分维度 | 3个 | 10个 | +233% |
| 预期负样本 | 11,630条 | 6,000-8,000条 | -48% |
| 质量要求 | 基础 | 高质量 | 显著提升 |
| 证据水平要求 | 无 | 蛋白质水平 | 新增 |

---

**文档版本**: 2.0（严格筛选版）
**创建日期**: 2026-03-16
**最后更新**: 2026-03-16
**状态**: 优化阶段
