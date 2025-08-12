import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')

# 1. 描述性统计分析
print("各裂变类型的转化情况：")
conversion_stats = df.groupby('裂变类型')['是否转化'].agg(['count', 'sum', 'mean'])
print(conversion_stats)

# 2. 卡方检验
# 创建一个交叉表
contingency_table = pd.crosstab(df['裂变类型'], df['是否转化'])
print("\n转化情况交叉表：")
print(contingency_table)

# 执行卡方检验
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\n卡方检验结果：Chi2 = {chi2:.4f}, p-value = {p:.4e}")

# 3. 可视化
# 设置图表风格
sns.set(style="whitegrid")

# 绘制转化率柱状图
plt.figure(figsize=(10, 6))
ax1 = sns.barplot(x='裂变类型', y='mean', data=conversion_stats.reset_index(), palette="viridis")
plt.title('不同裂变类型的转化率', fontsize=16)
plt.xlabel('裂变类型', fontsize=12)
plt.ylabel('转化率', fontsize=12)
# 在柱状图上添加数值标签
for i, v in enumerate(conversion_stats['mean']):
    ax1.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/01_裂变策略效果评估/conversion_rate.png')
plt.close()

# 绘制堆叠柱状图
plt.figure(figsize=(10, 6))
# 重新组织数据用于堆叠柱状图
stacked_data = df.groupby(['裂变类型', '是否转化']).size().unstack(fill_value=0)
stacked_data.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(10, 6))
plt.title('不同裂变类型的转化分布', fontsize=16)
plt.xlabel('裂变类型', fontsize=12)
plt.ylabel('用户数', fontsize=12)
plt.xticks(rotation=0)
plt.legend(['未转化', '转化'])
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/01_裂变策略效果评估/conversion_distribution.png')
plt.close()

print("\n图表已生成并保存。")