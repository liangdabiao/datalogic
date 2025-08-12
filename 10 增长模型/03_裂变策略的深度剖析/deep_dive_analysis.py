import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')

# 创建一个字典来存储所有分析结果
analysis_results = {}

print("--- 开始裂变策略的深度剖析 ---")

# --- 3.1 多种裂变行为的协同效应分析 ---
print("\n1. 多种裂变行为的协同效应分析")

# 创建一个新字段，表示用户历史参与裂变活动的组合情况
# 000, 001, 010, 011, 100, 101, 110, 111
df['历史行为组合'] = df['曾助力'].astype(str) + df['曾拼团'].astype(str) + df['曾推荐'].astype(str)
# 可以给组合起个更易读的名字
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

# 计算不同组合用户群体的整体转化率
synergy_analysis = df.groupby('历史行为组合_名称')['是否转化'].agg(['count', 'sum', 'mean']).reset_index()
synergy_analysis.rename(columns={'mean': '转化率'}, inplace=True)
# 按转化率排序以便观察
synergy_analysis = synergy_analysis.sort_values(by='转化率', ascending=False)
analysis_results['synergy_analysis'] = synergy_analysis
print("\n按历史行为组合分组的转化率:")
print(synergy_analysis)


# --- 3.2 用户裂变策略偏好分析 (基于代理指标) ---
print("\n2. 用户裂变策略偏好分析 (基于代理指标)")

# 分析具有不同历史行为的用户在当前看到不同裂变页面时的转化率差异
# 构建偏好分析矩阵: 行为历史 x 当前策略
preference_matrix = df.groupby(['历史行为组合_名称', '裂变类型'])['是否转化'].mean().reset_index()
preference_matrix.rename(columns={'是否转化': '转化率'}, inplace=True)
analysis_results['preference_matrix'] = preference_matrix
print("\n用户历史行为组合 x 当前裂变类型的转化率 (偏好分析):")
print(preference_matrix.head(10)) # 打印前10行


# --- 3.3 裂变活动的累积/疲劳效应分析 ---
print("\n3. 裂变活动的累积/疲劳效应分析")

# 分析用户参与裂变活动的总次数与当前转化率的关系
df['历史活动总次数'] = df['曾助力'] + df['曾拼团'] + df['曾推荐']
# 分析不同次数用户的转化率
activity_count_analysis = df.groupby('历史活动总次数')['是否转化'].agg(['count', 'sum', 'mean']).reset_index()
activity_count_analysis.rename(columns={'mean': '转化率'}, inplace=True)
analysis_results['activity_count_analysis'] = activity_count_analysis
print("\n按历史活动总次数分组的转化率:")
print(activity_count_analysis)

# 分析用户在当前 `裂变类型` 与其历史行为相同时的表现
# 例如，分析曾参与过“助力”的用户，在看到“助力砍价”时的表现
# 我们可以创建一个字段来标记这种情况
df['当前策略_匹配历史'] = '不匹配'
df.loc[(df['曾助力'] == 1) & (df['裂变类型'] == '助力砍价'), '当前策略_匹配历史'] = '匹配_助力'
df.loc[(df['曾拼团'] == 1) & (df['裂变类型'] == '拼团狂买'), '当前策略_匹配历史'] = '匹配_拼团'
# 对于无裂变页面，我们定义为不匹配
# 对于未参与过任何活动的用户，也定义为不匹配

match_analysis = df.groupby('当前策略_匹配历史')['是否转化'].agg(['count', 'sum', 'mean']).reset_index()
match_analysis.rename(columns={'mean': '转化率'}, inplace=True)
# 重新排序以便观察
match_analysis['当前策略_匹配历史'] = pd.Categorical(match_analysis['当前策略_匹配历史'], categories=['不匹配', '匹配_助力', '匹配_拼团'])
match_analysis = match_analysis.sort_values('当前策略_匹配历史').reset_index(drop=True)
analysis_results['match_analysis'] = match_analysis
print("\n当前裂变策略是否匹配用户历史行为的转化率:")
print(match_analysis)


print("\n--- 裂变策略的深度剖析完成 ---")

# 保存部分关键结果到CSV文件
for key, result_df in analysis_results.items():
    result_df.to_csv(f'E:/datalogic-main/10 增长模型/03_裂变策略的深度剖析/{key}.csv', index=False, encoding='utf-8-sig')

print("\n关键分析结果已保存为 CSV 文件。")