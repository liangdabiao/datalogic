import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置中文字体 (更明确地指定字体文件路径)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

# --- 可视化: 不同激励系数下的ROI ---
# 读取ROI分析结果
roi_df = pd.read_csv('E:/datalogic-main/10 增长模型/04_成本效益初步思考/roi_analysis.csv')

plt.figure(figsize=(12, 6))
# 使用 seaborn 的 hue 参数来区分不同的裂变类型
barplot = sns.barplot(x='激励系数', y='ROI', hue='裂变类型', data=roi_df, palette="viridis")
plt.title('不同裂变策略在不同激励系数下的ROI估算', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('激励系数', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('ROI (投资回报率)', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.legend(title='裂变类型', title_fontsize=12, prop=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上添加数值标签
# 遍历每个柱子
# for container in barplot.containers:
#     barplot.bar_label(container, fmt='%.2f', padding=3)

plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/04_成本效益初步思考/roi_by_incentive.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 可视化: 各策略的总收入和总成本 (以15%激励为例) ---
# 读取基础统计数据
basic_stats = pd.read_csv('E:/datalogic-main/10 增长模型/04_成本效益初步思考/basic_stats.csv')

# 选择一个特定的激励率（例如15%）来展示成本和收入
incentive_rate_for_display = 15
cost_col = f'总成本_{incentive_rate_for_display}%'

# 准备数据
plot_data = basic_stats[['裂变类型', '总收入', cost_col]].melt(id_vars=['裂变类型'], 
                                          value_vars=['总收入', cost_col], 
                                          var_name='指标', value_name='金额')

plt.figure(figsize=(10, 6))
sns.barplot(x='裂变类型', y='金额', hue='指标', data=plot_data, palette=["#2ecc71", "#e74c3c"]) # 绿色代表收入，红色代表成本
plt.title(f'各裂变策略的总收入与总成本对比 (激励系数: {incentive_rate_for_display}%)', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('裂变类型', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('金额 (元)', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.legend(title='指标', title_fontsize=12, prop=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/04_成本效益初步思考/revenue_cost_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("可视化图表已生成并保存。")