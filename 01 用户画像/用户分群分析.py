import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = r'E:\datalogic-main\01 用户画像\特征分析后的数据.csv'
df = pd.read_csv(file_path, encoding='utf-8')

print("=== 开始进行用户分群分析 ===")

# RFM分析
# 定义R、F、M的分群标准
def r_score(x):
    if x <= 3:
        return '新用户'
    elif x <= 6:
        return '活跃'
    elif x <= 9:
        return '稳定'
    else:
        return '老用户'

def f_score(x):
    if x <= 3:
        return '低频'
    elif x <= 5:
        return '中频'
    else:
        return '高频'

def m_score(x):
    if x <= 100:
        return '低价值'
    elif x <= 200:
        return '中价值'
    else:
        return '高价值'

# 应用分群标准
df['R_Stage'] = df['已注册月'].apply(r_score)
df['F_Stage'] = df['下单次数'].apply(f_score)
df['M_Stage'] = df['年消费'].apply(m_score)

# 用户价值分群
def value_score(row):
    if row['收入等级'] == '高收入' and row['消费等级'] == '高消费':
        return '高价值用户'
    elif row['收入等级'] == '高收入' and row['消费等级'] == '低消费':
        return '潜力用户'
    elif row['收入等级'] == '中收入' and row['消费等级'] == '中消费':
        return '价值用户'
    elif row['收入等级'] == '低收入' and row['收入消费比'] > df['收入消费比'].median():
        return '价格敏感'
    else:
        return '其他'

df['用户价值'] = df.apply(value_score, axis=1)

# 用户生命周期分群
def life_cycle_score(x):
    if x <= 3:
        return '新用户'
    elif x <= 6:
        return '活跃用户'
    elif x <= 9:
        return '忠诚用户'
    else:
        return '老用户'

df['生命周期'] = df['已注册月'].apply(life_cycle_score)

# 产品偏好分群
# 直接使用"近期购买产品"作为产品偏好标签

# 性别与产品偏好交叉分析
gender_product = pd.crosstab(df['性别'], df['近期购买产品'])
print("\n=== 性别与产品偏好交叉分析 ===")
print(gender_product)

# 年龄与产品偏好交叉分析
age_product = pd.crosstab(df['年龄层次'], df['近期购买产品'])
print("\n=== 年龄与产品偏好交叉分析 ===")
print(age_product)

# 视力与产品偏好交叉分析
vision_product = pd.crosstab(df['视力'], df['近期购买产品'])
print("\n=== 视力与产品偏好交叉分析 ===")
print(vision_product)

# 综合分群
# 我们可以基于多个维度进行综合分群，例如：
# 1. 年轻护眼刚需族: Z世代 + 护眼类
# 2. 高收入美妆爱好者: 高收入 + 美妆类
# 3. 中年视力关爱群体: 千禧一代/成熟用户 + 护眼类/医疗类 + 视力>=4
# 4. 价格敏感学生党: Z世代 + 低收入 + 低消费
# 5. 忠诚高价值用户: 忠诚用户 + 高价值用户
# 6. 新兴潜力用户: 新用户 + 潜力用户
# 7. 流失风险用户: 老用户 + 低频

def comprehensive_segment(row):
    if row['年龄层次'] == 'Z世代(18-27)' and ('贝尔防蓝光眼镜' in row['近期购买产品'] or '敦乐视疲劳滴眼液' in row['近期购买产品']):
        return '年轻护眼刚需族'
    elif row['收入等级'] == '高收入' and '9色钻石珠光眼影盘' in row['近期购买产品']:
        return '高收入美妆爱好者'
    elif (row['年龄层次'] in ['千禧一代(28-38)', '成熟用户(39+)']) and ('贝尔防蓝光眼镜' in row['近期购买产品'] or '敦乐视疲劳滴眼液' in row['近期购买产品']) and row['视力'] >= 4:
        return '中年视力关爱群体'
    elif row['年龄层次'] == 'Z世代(18-27)' and row['收入等级'] == '低收入' and row['消费等级'] == '低消费':
        return '价格敏感学生党'
    elif row['生命周期'] == '忠诚用户' and row['用户价值'] == '高价值用户':
        return '忠诚高价值用户'
    elif row['生命周期'] == '新用户' and row['用户价值'] == '潜力用户':
        return '新兴潜力用户'
    elif row['生命周期'] == '老用户' and row['F_Stage'] == '低频':
        return '流失风险用户'
    else:
        return '其他'

df['综合分群'] = df.apply(comprehensive_segment, axis=1)

# 查看各分群的数量
segment_counts = df['综合分群'].value_counts()
print("\n=== 综合分群结果 ===")
print(segment_counts)

# 保存最终分群结果
df.to_csv(r'E:\datalogic-main\01 用户画像\用户分群结果.csv', index=False, encoding='utf-8')

print("\n=== 分群结果已保存至 '用户分群结果.csv' ===")

# 可视化部分
# 1. 综合分群分布
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='综合分群', order=df['综合分群'].value_counts().index)
plt.title('用户综合分群分布')
plt.xlabel('用户数量')
plt.ylabel('用户群体')
plt.tight_layout()
plt.savefig(r'E:\datalogic-main\01 用户画像\用户综合分群分布.png')
plt.show()

# 2. 各分群的平均年消费
plt.figure(figsize=(10, 6))
segment_monetary = df.groupby('综合分群')['年消费'].mean().sort_values(ascending=False)
sns.barplot(x=segment_monetary.values, y=segment_monetary.index)
plt.title('各用户群体平均年消费')
plt.xlabel('平均年消费 (元)')
plt.ylabel('用户群体')
plt.tight_layout()
plt.savefig(r'E:\datalogic-main\01 用户画像\各用户群体平均年消费.png')
plt.show()

# 3. 各分群的平均年收入
plt.figure(figsize=(10, 6))
segment_income = df.groupby('综合分群')['年收入'].mean().sort_values(ascending=False)
sns.barplot(x=segment_income.values, y=segment_income.index)
plt.title('各用户群体平均年收入')
plt.xlabel('平均年收入 (元)')
plt.ylabel('用户群体')
plt.tight_layout()
plt.savefig(r'E:\datalogic-main\01 用户画像\各用户群体平均年收入.png')
plt.show()

# 4. 产品偏好分布
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='近期购买产品')
plt.title('产品偏好分布')
plt.xlabel('产品')
plt.ylabel('用户数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'E:\datalogic-main\01 用户画像\产品偏好分布.png')
plt.show()

# 5. 性别与产品偏好的关系
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='近期购买产品', hue='性别')
plt.title('性别与产品偏好的关系')
plt.xlabel('产品')
plt.ylabel('用户数量')
plt.xticks(rotation=45)
plt.legend(title='性别')
plt.tight_layout()
plt.savefig(r'E:\datalogic-main\01 用户画像\性别与产品偏好的关系.png')
plt.show()

print("\n=== 所有图表已保存 ===")