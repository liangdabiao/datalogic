# 用户画像分析技能 - 使用示例

## 快速开始示例

### 示例1: 基础用户画像分析

假设你有一个电商用户数据文件 `users.csv`，想要进行基础的用户画像分析：

```bash
# 运行用户分群分析
python scripts/user_segmentation.py --data users.csv --output my_analysis

# 生成可视化图表
python scripts/visualization.py --data my_analysis/用户分群结果.csv --output my_charts
```

**预期输出:**
- `my_analysis/用户分群结果.csv` - 包含完整分群结果的数据文件
- `my_analysis/基础统计信息.txt` - 基础统计信息摘要
- `my_charts/` - 包含各种分析图表的文件夹

### 示例2: 自定义参数分析

如果你想要调整RFM分析的阈值：

```bash
python scripts/user_segmentation.py --data users.csv \
    --r-thresholds 2 4 6 \
    --f-thresholds 2 4 \
    --m-thresholds 80 150 \
    --output custom_analysis
```

**参数说明:**
- `--r-thresholds 2 4 6`: R分群阈值调整为新用户(≤2月)、活跃(3-4月)、稳定(5-6月)、老用户(≥7月)
- `--f-thresholds 2 4`: F分群阈值调整为低频(≤2次)、中频(3-4次)、高频(≥5次)
- `--m-thresholds 80 150`: M分群阈值调整为低价值(≤80元)、中价值(81-150元)、高价值(≥151元)

## 实际应用场景

### 场景1: 电商平台用户分群

**背景**: 一个电商平台希望了解其用户群体特征，制定精准营销策略。

**数据示例** (`ecommerce_users.csv`):
```csv
用户编号,年龄,性别,年收入,年消费,下单次数,已注册月,近期购买产品
U001,25,女,45000,280,4,6,9色钻石珠光眼影盘
U002,32,男,68000,450,7,12,贝尔防蓝光眼镜(高级黑)
U003,28,女,52000,180,3,4,敦乐视疲劳滴眼液(13ML)
U004,35,男,85000,680,9,8,贝尔防蓝光眼镜(高级黑)
U005,23,女,28000,95,2,3,9色钻石珠光眼影盘
```

**执行分析**:
```bash
python scripts/user_segmentation.py --data ecommerce_users.csv --output ecommerce_analysis
python scripts/visualization.py --data ecommerce_analysis/用户分群结果.csv --output ecommerce_charts
```

**分析结果应用**:
1. **识别高价值用户**: 针对"高收入美妆爱好者"推送高端美妆产品
2. **新用户激活**: 对"新兴潜力用户"发放新人优惠券
3. **流失预警**: 针对"流失风险用户"进行召回营销

### 场景2: 金融服务客户分析

**背景**: 一家金融公司希望分析客户价值，优化服务策略。

**数据示例** (`finance_clients.csv`):
```csv
客户ID,年龄,性别,年收入,年消费,交易次数,注册月数
C001,45,男,120000,15000,24,36
C002,38,女,85000,8500,18,24
C003,29,男,65000,4500,12,8
C004,52,女,150000,22000,36,48
C005,33,男,75000,6800,15,15
```

**调整分析参数**:
```bash
python scripts/user_segmentation.py --data finance_clients.csv \
    --r-thresholds 6 12 24 \
    --f-thresholds 10 20 \
    --m-thresholds 5000 10000 \
    --output finance_analysis
```

**分析重点**:
- 长期高价值客户的维护策略
- 中等收入客户的提升方案
- 新客户的快速激活

### 场景3: 教育平台学员分析

**背景**: 在线教育平台希望分析学员行为，制定个性化学习计划。

**数据示例** (`education_students.csv`):
```csv
学员ID,年龄,性别,年收入,年消费,课程次数,注册月数,学习等级
S001,22,女,35000,1200,8,4,初级
S002,26,男,48000,2400,12,9,中级
S003,19,女,15000,600,3,2,入门
S004,31,女,62000,3600,18,15,高级
S005,24,男,38000,1800,10,7,中级
```

## Python 集成示例

### 示例4: 在Jupyter Notebook中使用

```python
import sys
sys.path.append('path/to/skill/scripts')

from user_segmentation import UserProfileAnalyzer
from visualization import UserProfilerVisualizer

# 创建分析器实例
analyzer = UserProfileAnalyzer('your_data.csv', 'output_analysis')

# 运行分析
if analyzer.run_full_analysis():
    print("分析完成!")

    # 创建可视化器
    visualizer = UserProfilerVisualizer('output_analysis/用户分群结果.csv', 'output_charts')
    visualizer.run_visualization()

    # 查看分群结果
    df = analyzer.df
    print(df['综合分群'].value_counts())
```

### 示例5: 自定义分析流程

```python
from user_segmentation import UserProfileAnalyzer
import pandas as pd

# 加载并分析数据
analyzer = UserProfileAnalyzer('users.csv')
analyzer.load_data()
analyzer.preprocess_data()

# 自定义分群逻辑
def custom_segment(row):
    if row['年收入'] > 100000 and row['年消费'] > 5000:
        return 'VIP客户'
    elif row['年龄'] < 25:
        return '年轻用户'
    elif row['下单次数'] > 10:
        return '高频用户'
    else:
        return '普通用户'

analyzer.df['自定义分群'] = analyzer.df.apply(custom_segment, axis=1)

# 分析自定义分群结果
segment_stats = analyzer.df.groupby('自定义分群').agg({
    '年收入': 'mean',
    '年消费': 'mean',
    '下单次数': 'mean',
    '用户编号': 'count'
}).rename(columns={'用户编号': '用户数量'})

print(segment_stats)
```

## 高级应用示例

### 示例6: 批量分析多个数据集

```python
import os
from user_segmentation import UserProfileAnalyzer

def analyze_multiple_datasets(data_dir, output_base_dir):
    """批量分析多个数据集"""
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    results = {}

    for data_file in data_files:
        data_path = os.path.join(data_dir, data_file)
        output_dir = os.path.join(output_base_dir, data_file.replace('.csv', '_analysis'))

        print(f"正在分析: {data_file}")

        analyzer = UserProfileAnalyzer(data_path, output_dir)
        if analyzer.run_full_analysis():
            # 保存关键统计信息
            stats = analyzer.generate_basic_stats()
            results[data_file] = stats
            print(f"✅ {data_file} 分析完成")
        else:
            print(f"❌ {data_file} 分析失败")

    return results

# 使用示例
results = analyze_multiple_datasets('data/', 'analysis_results/')

# 输出汇总报告
for file, stats in results.items():
    print(f"\n=== {file} ===")
    print(f"总用户数: {stats['总用户数']}")
    print(f"平均年消费: {stats['平均年消费']:.2f}")
```

### 示例7: 生成分析报告

```python
from user_segmentation import UserProfileAnalyzer
from datetime import datetime

def generate_analysis_report(data_path, output_file):
    """生成详细的分析报告"""
    analyzer = UserProfileAnalyzer(data_path)
    analyzer.run_full_analysis()

    df = analyzer.df
    stats = analyzer.generate_basic_stats()

    # 生成报告内容
    report = f"""# 用户画像分析报告

## 基本信息
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据源: {data_path}
- 总用户数: {stats['总用户数']:,}

## 用户特征概览
- 平均年龄: {stats['平均年龄']:.1f}岁
- 平均年收入: {stats['平均年收入']:,.0f}元
- 平均年消费: {stats['平均年消费']:.1f}元
- 平均下单次数: {stats['平均下单次数']:.1f}次
- 平均注册月数: {stats['平均注册月数']:.1f}个月

## 性别分布
"""

    for gender, count in stats['性别分布'].items():
        report += f"- {gender}: {count}人 ({count/stats['总用户数']*100:.1f}%)\n"

    report += "\n## 收入分布\n"
    for level, count in stats['收入分布'].items():
        report += f"- {level}: {count}人 ({count/stats['总用户数']*100:.1f}%)\n"

    report += "\n## 用户分群结果\n"
    for segment, count in stats['用户分群分布'].items():
        report += f"- {segment}: {count}人 ({count/stats['总用户数']*100:.1f}%)\n"

    report += "\n## 详细分群分析\n"

    # 每个分群的详细分析
    segment_analysis = df.groupby('综合分群').agg({
        '年收入': ['mean', 'std'],
        '年消费': ['mean', 'std'],
        '下单次数': 'mean',
        '年龄': 'mean',
        '用户编号': 'count'
    }).round(2)

    for segment in segment_analysis.index:
        data = segment_analysis.loc[segment]
        report += f"\n### {segment}\n"
        report += f"- 用户数量: {data[('用户编号', 'count')]}人\n"
        report += f"- 平均年收入: {data[('年收入', 'mean')]:,.0f}元\n"
        report += f"- 平均年消费: {data[('年消费', 'mean')]:,.1f}元\n"
        report += f"- 平均下单次数: {data[('下单次数', 'mean')]:.1f}次\n"
        report += f"- 平均年龄: {data[('年龄', 'mean')]:.1f}岁\n"

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {output_file}")

# 使用示例
generate_analysis_report('users.csv', '用户画像分析报告.md')
```

## 数据准备示例

### 示例8: 从多个数据源合并

```python
import pandas as pd
from user_segmentation import UserProfileAnalyzer

# 假设有多个数据源需要合并
def merge_user_data(user_info_file, transaction_file, product_file):
    """合并多个用户数据源"""

    # 加载基础用户信息
    user_info = pd.read_csv(user_info_file)

    # 加载交易数据并聚合
    transactions = pd.read_csv(transaction_file)
    trans_summary = transactions.groupby('用户编号').agg({
        '消费金额': 'sum',
        '订单编号': 'count',
        '订单日期': ['min', 'max']
    }).reset_index()

    # 重命名列
    trans_summary.columns = ['用户编号', '年消费', '下单次数', '首次消费', '最近消费']

    # 加载产品信息
    products = pd.read_csv(product_file)

    # 合并数据
    merged_data = user_info.merge(trans_summary, on='用户编号', how='left')

    # 计算注册时长
    merged_data['已注册月'] = 12  # 假设数据
    merged_data['近期购买产品'] = '未知'  # 可以从产品数据中获取

    # 填充缺失值
    merged_data.fillna({
        '年消费': 0,
        '下单次数': 0
    }, inplace=True)

    return merged_data

# 使用示例
combined_data = merge_user_data('user_info.csv', 'transactions.csv', 'products.csv')

# 保存合并后的数据
combined_data.to_csv('combined_user_data.csv', index=False, encoding='utf-8-sig')

# 运行分析
analyzer = UserProfileAnalyzer('combined_user_data.csv')
analyzer.run_full_analysis()
```

### 示例9: 数据清洗和预处理

```python
import pandas as pd
import numpy as np

def clean_user_data(raw_data_path, clean_data_path):
    """数据清洗和预处理"""

    # 读取原始数据
    df = pd.read_csv(raw_data_path, encoding='gbk')

    print(f"原始数据形状: {df.shape}")

    # 1. 处理重复数据
    df = df.drop_duplicates()
    print(f"去重后数据形状: {df.shape}")

    # 2. 处理缺失值
    # 数值型字段用中位数填充
    numeric_cols = ['年龄', '年收入', '年消费', '下单次数', '已注册月']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # 分类字段用众数填充
    categorical_cols = ['性别', '近期购买产品']
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else '未知'
            df[col].fillna(mode_val, inplace=True)

    # 3. 处理异常值
    # 年龄异常值处理 (18-80岁)
    df['年龄'] = df['年龄'].clip(18, 80)

    # 收入异常值处理 (使用IQR方法)
    if '年收入' in df.columns:
        Q1 = df['年收入'].quantile(0.25)
        Q3 = df['年收入'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['年收入'] = df['年收入'].clip(lower_bound, upper_bound)

    # 消费异常值处理 (非负数)
    if '年消费' in df.columns:
        df['年消费'] = df['年消费'].clip(lower=0)

    # 4. 数据类型优化
    dtype_mapping = {
        '年龄': 'int8',
        '性别': 'category',
        '年收入': 'int32',
        '年消费': 'float32',
        '下单次数': 'int8',
        '已注册月': 'int8'
    }

    for col, dtype in dtype_mapping.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # 5. 标准化分类字段
    if '性别' in df.columns:
        df['性别'] = df['性别'].str.strip().str.title()
        df['性别'] = df['性别'].replace({'M': '男', 'F': '女'})

    print(f"清洗后数据形状: {df.shape}")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # 保存清洗后的数据
    df.to_csv(clean_data_path, index=False, encoding='utf-8-sig')
    print(f"清洗后数据已保存到: {clean_data_path}")

    return df

# 使用示例
clean_data = clean_user_data('raw_users.csv', 'clean_users.csv')

# 对清洗后的数据进行分析
from user_segmentation import UserProfileAnalyzer
analyzer = UserProfileAnalyzer('clean_users.csv')
analyzer.run_full_analysis()
```

这些示例展示了用户画像分析技能在不同场景下的应用方式，从基础使用到高级定制，为实际业务需求提供了灵活的解决方案。