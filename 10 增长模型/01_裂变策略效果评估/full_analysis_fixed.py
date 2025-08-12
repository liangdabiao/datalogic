import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 尝试多种方式设置中文字体
import matplotlib.font_manager as fm

# 查找系统中可用的中文字体
# chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'Sim' in f.name or 'Kai' in f.name or 'Fang' in f.name]
# print(chinese_fonts)

# 明确指定一个常见的中文字体，例如 SimHei (黑体)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

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
# 强制设置字体以确保标题和轴标签正确显示
plt.gca().set_title('不同裂变类型的转化率', fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=16))
plt.gca().set_xlabel('裂变类型', fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12))
plt.gca().set_ylabel('转化率', fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/01_裂变策略效果评估/conversion_rate_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制堆叠柱状图
plt.figure(figsize=(10, 6))
# 重新组织数据用于堆叠柱状图
stacked_data = df.groupby(['裂变类型', '是否转化']).size().unstack(fill_value=0)
ax2 = stacked_data.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], figsize=(10, 6))
plt.title('不同裂变类型的转化分布', fontsize=16)
plt.xlabel('裂变类型', fontsize=12)
plt.ylabel('用户数', fontsize=12)
plt.xticks(rotation=0)
plt.legend(['未转化', '转化'], prop=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
# 强制设置字体
plt.gca().set_title('不同裂变类型的转化分布', fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=16))
plt.gca().set_xlabel('裂变类型', fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12))
plt.gca().set_ylabel('用户数', fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/01_裂变策略效果评估/conversion_distribution_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n图表已生成并保存 (已尝试修复中文显示)。")