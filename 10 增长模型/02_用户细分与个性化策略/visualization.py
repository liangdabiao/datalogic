import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置中文字体 (更明确地指定字体文件路径)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')

# 重新进行一些关键的交叉分析并可视化

# --- 可视化 1: 设备 x 裂变类型 ---
plt.figure(figsize=(10, 6))
sns.barplot(x='设备', y='是否转化', hue='裂变类型', data=df, palette="viridis")
plt.title('不同设备用户对各裂变策略的转化率', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('设备类型', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('转化率', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.legend(title='裂变类型', title_fontsize=12, prop=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/02_用户细分与个性化策略/device_fission_conversion.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 可视化 2: 城市类型 x 裂变类型 ---
plt.figure(figsize=(10, 6))
sns.barplot(x='城市类型', y='是否转化', hue='裂变类型', data=df, palette="viridis")
plt.title('不同城市用户对各裂变策略的转化率', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('城市类型', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('转化率', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.legend(title='裂变类型', title_fontsize=12, prop=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/02_用户细分与个性化策略/city_fission_conversion.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 可视化 3: RFM 矩阵热力图 ---
# 根据 R值 和 M值 进行分组
df['R_等级'] = pd.qcut(df['R值'], q=3, labels=['低活跃度', '中活跃度', '高活跃度'])
df['M_等级'] = pd.qcut(df['M值'], q=3, labels=['低消费', '中消费', '高消费'])
# 计算 RFM 矩阵
rfm_matrix_pivot = df.groupby(['R_等级', 'M_等级'])['是否转化'].mean().unstack()

plt.figure(figsize=(8, 6))
sns.heatmap(rfm_matrix_pivot, annot=True, fmt=".4f", cmap="Blues", linewidths=.5)
plt.title('RFM 矩阵：不同用户价值群体的平均转化率', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('消费金额等级 (M)', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('活跃度等级 (R)', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/02_用户细分与个性化策略/rfm_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 可视化 4: 历史行为 x 裂变类型 (以曾助力为例) ---
plt.figure(figsize=(10, 6))
sns.barplot(x='曾助力', y='是否转化', hue='裂变类型', data=df, palette="viridis")
plt.title('是否曾参与助力活动的用户对各裂变策略的转化率', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('是否曾参与助力 (0: 否, 1: 是)', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('转化率', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.legend(title='裂变类型', title_fontsize=12, prop=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/02_用户细分与个性化策略/history_fission_conversion.png', dpi=300, bbox_inches='tight')
plt.close()

print("可视化图表已生成并保存。")