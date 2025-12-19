# 用户画像分析技能 - 详细参考文档

## API 文档

### UserProfileAnalyzer 类

主要的用户画像分析类，提供完整的分析流程。

#### 构造函数

```python
UserProfileAnalyzer(data_path, output_dir="analysis_output")
```

**参数:**
- `data_path` (str): 数据文件路径
- `output_dir` (str): 输出目录，默认为 "analysis_output"

#### 主要方法

##### load_data()
```python
load_data() -> bool
```
加载CSV数据文件。

**返回值:** bool - 成功返回True，失败返回False

##### validate_data()
```python
validate_data() -> bool
```
验证数据格式，检查必要的列是否存在。

**必需列:** 用户编号、年龄、性别、年收入、年消费、下单次数、已注册月

##### preprocess_data()
```python
preprocess_data() -> None
```
数据预处理，包括：
- 缺失值处理
- 数据类型转换
- 衍生字段创建（收入消费比、年龄层次、收入等级、消费等级）

##### rfm_analysis()
```python
rfm_analysis(r_thresholds=(3, 6, 9), f_thresholds=(3, 5), m_thresholds=(100, 200)) -> None
```

**参数:**
- `r_thresholds` (tuple): R分群阈值，默认(3, 6, 9)
- `f_thresholds` (tuple): F分群阈值，默认(3, 5)
- `m_thresholds` (tuple): M分群阈值，默认(100, 200)

**分群逻辑:**
- **R分群 (注册时长):** 新用户(≤3月) → 活跃(4-6月) → 稳定(7-9月) → 老用户(≥10月)
- **F分群 (下单频率):** 低频(≤3次) → 中频(4-5次) → 高频(≥6次)
- **M分群 (消费金额):** 低价值(≤100元) → 中价值(101-200元) → 高价值(≥201元)

##### value_analysis()
```python
value_analysis() -> None
```
用户价值分析，基于收入等级和消费等级进行用户价值分类。

**用户价值分类:**
- 高价值用户: 高收入 + 高消费
- 潜力用户: 高收入 + 低消费
- 价值用户: 中收入 + 中消费
- 价格敏感: 低收入 + 高收入消费比

##### comprehensive_segmentation()
```python
comprehensive_segmentation() -> None
```
综合用户分群，结合多个维度识别7大核心用户群体：

1. **年轻护眼刚需族**: Z世代(18-27) + 护眼类产品
2. **高收入美妆爱好者**: 高收入 + 美妆类产品
3. **中年视力关爱群体**: 28岁+ + 护眼产品 + 视力问题
4. **价格敏感学生党**: 年轻 + 低收入 + 低消费
5. **忠诚高价值用户**: 长期活跃 + 高消费
6. **新兴潜力用户**: 新注册 + 高收入潜力
7. **流失风险用户**: 老用户 + 低频购买

##### run_full_analysis()
```python
run_full_analysis(r_thresholds=(3, 6, 9), f_thresholds=(3, 5), m_thresholds=(100, 200)) -> bool
```
运行完整的分析流程。

**返回值:** bool - 成功返回True，失败返回False

### UserProfilerVisualizer 类

用户画像可视化类，生成各种分析图表。

#### 构造函数

```python
UserProfilerVisualizer(data_path, output_dir="visualization_output")
```

#### 主要方法

##### plot_segment_distribution()
生成用户分群分布的横向条形图。

##### plot_consumption_by_segment()
生成各用户群体平均消费对比图。

##### plot_income_by_segment()
生成各用户群体平均收入对比图。

##### plot_demographic_distribution()
生成人口统计学特征分布图（性别、年龄、收入等）。

##### plot_rfm_analysis()
生成RFM分析结果的饼图。

##### plot_cross_analysis()
生成用户特征交叉分析热力图。

##### plot_income_consumption_scatter()
生成收入-消费关系散点图。

##### generate_summary_dashboard()
生成综合分析仪表板，包含所有关键指标的汇总视图。

##### run_visualization()
```python
run_visualization() -> bool
```
运行所有可视化流程。

## 数据格式要求

### 输入数据格式

**文件格式:** CSV，UTF-8编码

**必需字段:**

| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| 用户编号 | 字符串 | 唯一用户标识 | "U001" |
| 年龄 | 整数 | 用户年龄 | 25 |
| 性别 | 字符串 | 用户性别 | "男" / "女" |
| 年收入 | 整数 | 年收入(元) | 50000 |
| 年消费 | 整数 | 年消费金额(元) | 1200 |
| 下单次数 | 整数 | 年度下单频次 | 5 |
| 已注册月 | 整数 | 注册时长(月) | 8 |

**可选字段:**

| 字段名 | 类型 | 描述 |
|--------|------|------|
| 近期购买产品 | 字符串 | 最近购买的商品名称 |
| 视力 | 整数 | 视力状况等级(1-5) |
| 状态 | 字符串 | 情感状态 |

### 输出数据格式

**分群结果文件:** `用户分群结果.csv`

**新增字段:**

| 字段名 | 描述 |
|--------|------|
| 收入消费比 | 年消费 / 年收入 |
| 年龄层次 | Z世代(18-27) / 千禧一代(28-38) / 成熟用户(39+) |
| 收入等级 | 低收入(≤3万) / 中收入(3-6万) / 高收入(>6万) |
| 消费等级 | 低消费(≤100) / 中消费(101-200) / 高消费(≥201) |
| R_Stage | RFM分析中的Recency分群 |
| F_Stage | RFM分析中的Frequency分群 |
| M_Stage | RFM分析中的Monetary分群 |
| 用户价值 | 用户价值分类 |
| 生命周期 | 新用户/活跃用户/忠诚用户/老用户 |
| 综合分群 | 最终的用户群体分类 |

## 命令行工具

### 用户分群分析

```bash
python scripts/user_segmentation.py --data your_data.csv --output analysis_output
```

**参数:**
- `--data`: 数据文件路径 (必需)
- `--output`: 输出目录 (默认: analysis_output)
- `--r-thresholds`: R分群阈值 (默认: 3 6 9)
- `--f-thresholds`: F分群阈值 (默认: 3 5)
- `--m-thresholds`: M分群阈值 (默认: 100 200)

**示例:**
```bash
# 使用默认参数
python scripts/user_segmentation.py --data users.csv

# 自定义阈值
python scripts/user_segmentation.py --data users.csv \
    --r-thresholds 2 5 8 \
    --f-thresholds 2 4 \
    --m-thresholds 80 150
```

### 可视化生成

```bash
python scripts/visualization.py --data analysis_output/用户分群结果.csv --output charts
```

**参数:**
- `--data`: 分析结果数据文件路径 (必需)
- `--output`: 图表输出目录 (默认: visualization_output)

## 高级用法

### 自定义分群逻辑

可以通过修改 `comprehensive_segmentation()` 方法来自定义用户分群逻辑：

```python
def custom_segmentation(row):
    # 自定义分群逻辑
    if condition1:
        return '自定义群体1'
    elif condition2:
        return '自定义群体2'
    else:
        return '其他群体'

# 应用自定义分群
df['自定义分群'] = df.apply(custom_segmentation, axis=1)
```

### 添加新的可视化

```python
def plot_custom_analysis(self):
    """自定义分析图表"""
    plt.figure(figsize=(10, 6))
    # 自定义绘图逻辑
    plt.title('自定义分析')
    plt.savefig(f'{self.output_dir}/自定义分析.png')
    plt.close()
```

### 集成其他分析模型

```python
from sklearn.cluster import KMeans

def add_ml_clustering(self, n_clusters=5):
    """添加机器学习聚类"""
    features = self.df[['年收入', '年消费', '下单次数', '已注册月']]

    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    self.df['ML分群'] = kmeans.fit_predict(features_scaled)
```

## 性能优化建议

### 大数据处理

对于大数据集（>10万用户）：

1. **使用数据采样:**
```python
# 随机采样10%数据进行分析
df_sample = self.df.sample(frac=0.1, random_state=42)
```

2. **优化内存使用:**
```python
# 使用合适的数据类型
dtypes = {
    '年龄': 'int8',
    '性别': 'category',
    '年收入': 'int32',
    '年消费': 'int32'
}
df = pd.read_csv(file_path, dtype=dtypes)
```

3. **并行处理:**
```python
from multiprocessing import Pool
# 使用多进程处理大数据集
```

### 可视化优化

对于大量数据点的散点图：

```python
# 使用透明度和采样避免过度绘制
plt.scatter(df['年收入'], df['年消费'], alpha=0.1, s=1)

# 或者使用hexbin图
plt.hexbin(df['年收入'], df['年消费'], gridsize=30, cmap='YlOrRd')
```

## 故障排除

### 常见问题

1. **中文字体显示问题**
```python
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

2. **数据编码问题**
```python
# 尝试不同的编码
encodings = ['utf-8', 'gbk', 'utf-8-sig']
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
```

3. **内存不足**
```python
# 分块处理大文件
chunk_size = 10000
chunks = pd.read_csv(large_file, chunksize=chunk_size)
for chunk in chunks:
    # 处理每个数据块
    process_chunk(chunk)
```

### 调试技巧

1. **查看数据信息**
```python
df.info()
df.describe()
df.isnull().sum()
```

2. **验证分析结果**
```python
# 检查分群结果
print(df['综合分群'].value_counts())

# 检查异常值
print(df[df['年收入'] < 0])
```

3. **日志记录**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"数据处理完成: {len(df)} 条记录")
```