import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = r'E:\datalogic-main\01 用户画像\爆款产品.csv'
df = pd.read_csv(file_path, encoding='utf-8', skiprows=1)

# 列名清洗
col_names = ['用户编号', '年龄', '性别', '状态', '下单次数', '视力', 
             '年收入', '年消费', '近期购买产品', '已注册月']
df.columns = col_names

# 保存清洗后的数据
df.to_csv(r'E:\datalogic-main\01 用户画像\清理后的用户数据.csv', index=False, encoding='utf-8')

print('=== 数据探索性分析结果 ===')
print()

# 数值型字段统计
numeric_cols = ['年龄', '下单次数', '视力', '年收入', '年消费', '已注册月']
for col in numeric_cols:
    stats = df[col].describe()
    print(f'【{col}】')
    print(f'  数量: {int(stats["count"])}')
    print(f'  均值: {stats["mean"]:.1f}')
    print(f'  中位数: {stats["50%"]:.0f}')
    print(f'  标准差: {stats["std"]:.1f}')
    print(f'  范围: {stats["min"]:.0f} - {stats["max"]:.0f}')
    print()

# 分类型字段
print('=== 分类型分布 ===')
cat_stats = {}
for col in ['性别', '状态', '近期购买产品']:
    counts = df[col].value_counts()
    cat_stats[col] = counts
    print(f'【{col}】')
    for val, count in counts.items():
        percent = count/len(df)*100
        print(f'  {val}: {count}人 ({percent:.1f}%)')
    print()

# 收入消费比分析
df['收入消费比'] = df['年消费'] / df['年收入'] * 100
print('=== 消费行为特征 ===')
print(f'平均收入消费比: {df["收入消费比"].mean():.1f}%')
print(f'高消费群体(年消费>200元): {len(df[df["年消费"]>200])}/{len(df)} ({len(df[df["年消费"]>200])/len(df)*100:.1f}%)')
print(f'高频消费群体(下单>4次): {len(df[df["下单次数"]>4])}/{len(df)} ({len(df[df["下单次数"]>4])/len(df)*100:.1f}%)')

# 年龄分群
df['年龄层次'] = pd.cut(df['年龄'], 
                      bins=[0, 27, 38, 100], 
                      labels=['Z世代(18-27)', '千禧一代(28-38)', '成熟用户(39+)'])

age_group = df['年龄层次'].value_counts()
print()
print('=== 年龄层次分布 ===')
for age, count in age_group.items():
    percent = count/len(df)*100
    print(f'{age}: {count}人 ({percent:.1f}%)')

# 用户价值分群
df['消费等级'] = pd.qcut(df['年消费'], q=3, labels=['低消费', '中消费', '高消费'])
df['收入等级'] = pd.qcut(df['年收入'], q=3, labels=['低收入', '中收入', '高收入'])

# 保存分析结果
df.to_csv(r'E:\datalogic-main\01 用户画像\特征分析后的数据.csv', index=False, encoding='utf-8')

print()
print(f'=== 数据文件保存完成 ===')
print('清理后数据: 清理后的用户数据.csv')
print('特征分析数据: 特征分析后的数据.csv')