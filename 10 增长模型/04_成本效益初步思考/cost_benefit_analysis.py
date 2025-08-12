import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')

print("--- 开始成本效益初步思考分析 ---")

# --- 准备基础数据 ---
# 按裂变类型分组，计算基础指标
grouped = df.groupby('裂变类型')

basic_stats = pd.DataFrame({
    '用户总数': grouped['用户码'].count(),
    '转化用户数': grouped['是否转化'].sum(),
    '转化率': grouped['是否转化'].mean()
}).reset_index()

# 计算转化用户的平均M值 (客单价代理)
converted_df = df[df['是否转化'] == 1]
avg_m_by_type = converted_df.groupby('裂变类型')['M值'].mean()
basic_stats = basic_stats.merge(avg_m_by_type.reset_index().rename(columns={'M值': '客单价_代理'}), on='裂变类型')

# --- 3.1 基于M值的用户价值评估 (总收入估算) ---
basic_stats['总收入'] = basic_stats['转化用户数'] * basic_stats['客单价_代理']
print("\n各裂变类型的基础数据:")
print(basic_stats)

# --- 3.2 & 3.3 潜在成本假设与ROI估算 ---
# 定义不同的激励系数进行模拟
incentive_rates = [0.05, 0.10, 0.15, 0.20, 0.25]

# 创建一个列表来存储不同情景下的结果
roi_summary = []

for rate in incentive_rates:
    col_cost = f'总成本_{int(rate*100)}%'
    col_roi = f'ROI_{int(rate*100)}%'
    
    # 总成本 = 转化用户数 * 客单价 * 激励系数
    basic_stats[col_cost] = basic_stats['转化用户数'] * basic_stats['客单价_代理'] * rate
    
    # ROI = (总收入 - 总成本) / 总成本
    basic_stats[col_roi] = (basic_stats['总收入'] - basic_stats[col_cost]) / basic_stats[col_cost]
    
    # 提取用于可视化的数据
    temp_df = basic_stats[['裂变类型', col_roi]].copy()
    temp_df.rename(columns={col_roi: 'ROI'}, inplace=True)
    temp_df['激励系数'] = f'{int(rate*100)}%'
    roi_summary.append(temp_df)

# 合并所有情景的结果
roi_df = pd.concat(roi_summary, ignore_index=True)

print("\n各裂变类型的总收入估算 (转化用户数 * 客单价):")
print(basic_stats[['裂变类型', '总收入']])

print("\n不同激励系数下的ROI估算结果 (示例: 10%激励):")
print(basic_stats[['裂变类型', 'ROI_10%']])

# 保存基础数据和ROI结果
basic_stats.to_csv('E:/datalogic-main/10 增长模型/04_成本效益初步思考/basic_stats.csv', index=False, encoding='utf-8-sig')
roi_df.to_csv('E:/datalogic-main/10 增长模型/04_成本效益初步思考/roi_analysis.csv', index=False, encoding='utf-8-sig')

# --- 3.4 敏感性分析 (准备数据) ---
# 准备用于敏感性分析的数据透视表
roi_pivot = roi_df.pivot(index='裂变类型', columns='激励系数', values='ROI')
roi_pivot.to_csv('E:/datalogic-main/10 增长模型/04_成本效益初步思考/roi_pivot.csv', encoding='utf-8-sig')

print("\nROI透视表已保存。")
print("\n--- 成本效益初步思考分析完成 ---")