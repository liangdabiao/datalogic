import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置中文字体 (更明确地指定字体文件路径)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')

# --- 可视化 1: 历史行为组合与转化率 (协同效应) ---
# 重新计算数据以确保正确性
df['历史行为组合'] = df['曾助力'].astype(str) + df['曾拼团'].astype(str) + df['曾推荐'].astype(str)
combination_names = {
    '000': '无历史行为',
    '001': '仅推荐',
    '010': '仅拼团',
    '011': '拼团+推荐',
    '100': '仅助力',
    '101': '助力+推荐',
    '110': '助力+拼团',
    '111': '全部参与'
}
df['历史行为组合_名称'] = df['历史行为组合'].map(combination_names)

synergy_data = df.groupby('历史行为组合_名称')['是否转化'].mean().reset_index()
synergy_data.rename(columns={'是否转化': '转化率'}, inplace=True)
# 按转化率排序
synergy_data = synergy_data.sort_values(by='转化率', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='历史行为组合_名称', y='转化率', data=synergy_data, palette="viridis")
plt.title('不同历史行为组合用户的转化率', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('历史行为组合', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('转化率', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/03_裂变策略的深度剖析/synergy_analysis.png', dpi=300, bbox_inches='tight')
plt.close()


# --- 可视化 2: 用户偏好矩阵热力图 ---
preference_matrix_pivot = df.groupby(['历史行为组合_名称', '裂变类型'])['是否转化'].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(preference_matrix_pivot, annot=True, fmt=".4f", cmap="Blues", linewidths=.5)
plt.title('用户偏好矩阵：不同历史行为用户对各裂变策略的转化率', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('当前裂变类型', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('历史行为组合', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/03_裂变策略的深度剖析/preference_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 可视化 3: 历史活动总次数与转化率 ---
df['历史活动总次数'] = df['曾助力'] + df['曾拼团'] + df['曾推荐']
activity_count_data = df.groupby('历史活动总次数')['是否转化'].mean().reset_index()
activity_count_data.rename(columns={'是否转化': '转化率'}, inplace=True)

plt.figure(figsize=(10, 6))
sns.lineplot(x='历史活动总次数', y='转化率', data=activity_count_data, marker='o')
plt.title('用户历史参与裂变活动总次数与转化率的关系', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('历史活动总次数', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('转化率', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xticks(activity_count_data['历史活动总次数'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/03_裂变策略的深度剖析/activity_count_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 可视化 4: 策略匹配与转化率 ---
df['当前策略_匹配历史'] = '不匹配'
df.loc[(df['曾助力'] == 1) & (df['裂变类型'] == '助力砍价'), '当前策略_匹配历史'] = '匹配_助力'
df.loc[(df['曾拼团'] == 1) & (df['裂变类型'] == '拼团狂买'), '当前策略_匹配历史'] = '匹配_拼团'

match_data = df.groupby('当前策略_匹配历史')['是否转化'].mean().reset_index()
match_data.rename(columns={'是否转化': '转化率'}, inplace=True)
# 重新排序
match_data['当前策略_匹配历史'] = pd.Categorical(match_data['当前策略_匹配历史'], categories=['不匹配', '匹配_助力', '匹配_拼团'])
match_data = match_data.sort_values('当前策略_匹配历史').reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='当前策略_匹配历史', y='转化率', data=match_data, palette="viridis")
plt.title('当前裂变策略与用户历史行为匹配度对转化率的影响', fontsize=16, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.xlabel('策略匹配情况', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.ylabel('转化率', fontsize=12, fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'))
plt.tight_layout()
plt.savefig('E:/datalogic-main/10 增长模型/03_裂变策略的深度剖析/match_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("可视化图表已生成并保存。")