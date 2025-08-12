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

print("--- 开始用户细分与个性化策略分析 ---")

# --- 3.1 基于用户基础属性的转化率分析 ---
print("\n1. 基于用户基础属性的转化率分析")

# 设备类型分析
device_conversion = df.groupby('设备')['是否转化'].agg(['count', 'sum', 'mean']).reset_index()
device_conversion.rename(columns={'mean': '转化率'}, inplace=True)
analysis_results['device_conversion'] = device_conversion
print("\n按设备类型分组的转化率:")
print(device_conversion)

# 城市类型分析
city_conversion = df.groupby('城市类型')['是否转化'].agg(['count', 'sum', 'mean']).reset_index()
city_conversion.rename(columns={'mean': '转化率'}, inplace=True)
analysis_results['city_conversion'] = city_conversion
print("\n按城市类型分组的转化率:")
print(city_conversion)

# --- 3.2 基于 RFM 概念的用户价值分析 ---
print("\n2. 基于 RFM 概念的用户价值分析")

# 查看 R值 和 M值 的分布
print("\nR值 (活跃度) 描述性统计:")
print(df['R值'].describe())
print("\nM值 (消费金额) 描述性统计:")
print(df['M值'].describe())

# 根据 R值 和 M值 进行分组
# 使用四分位数作为分界点是一种常见的方法
df['R_等级'] = pd.qcut(df['R值'], q=3, labels=['低活跃度', '中活跃度', '高活跃度'])
df['M_等级'] = pd.qcut(df['M值'], q=3, labels=['低消费', '中消费', '高消费'])

# 创建 RFM 矩阵
rfm_matrix = df.groupby(['R_等级', 'M_等级'])['是否转化'].mean().reset_index()
rfm_matrix.rename(columns={'是否转化': '转化率'}, inplace=True)
analysis_results['rfm_matrix'] = rfm_matrix
print("\nRFM 矩阵 (按 R_等级 和 M_等级 分组的转化率):")
print(rfm_matrix)

# --- 3.3 历史行为影响分析 ---
print("\n3. 历史行为影响分析")

# 定义一个函数来简化分析
def analyze_history_behavior(column_name, description):
    behavior_conversion = df.groupby(column_name)['是否转化'].agg(['count', 'sum', 'mean']).reset_index()
    behavior_conversion.rename(columns={'mean': '转化率'}, inplace=True)
    analysis_results[f'{column_name}_conversion'] = behavior_conversion
    print(f"\n按 {description} 分组的转化率:")
    print(behavior_conversion)

analyze_history_behavior('曾助力', '是否曾参与助力')
analyze_history_behavior('曾拼团', '是否曾参与拼团')
analyze_history_behavior('曾推荐', '是否曾参与推荐')

# --- 3.4 多维度交叉分析 ---
print("\n4. 多维度交叉分析")

# 设备 x 裂变类型
device_fission_conversion = df.groupby(['设备', '裂变类型'])['是否转化'].mean().reset_index()
device_fission_conversion.rename(columns={'是否转化': '转化率'}, inplace=True)
analysis_results['device_fission_conversion'] = device_fission_conversion
print("\n设备 x 裂变类型的转化率:")
print(device_fission_conversion.head(10)) # 打印前10行以节省空间

# 城市类型 x 裂变类型
city_fission_conversion = df.groupby(['城市类型', '裂变类型'])['是否转化'].mean().reset_index()
city_fission_conversion.rename(columns={'是否转化': '转化率'}, inplace=True)
analysis_results['city_fission_conversion'] = city_fission_conversion
print("\n城市类型 x 裂变类型的转化率:")
print(city_fission_conversion.head(10))

# RFM分组 x 裂变类型
rf_fission_conversion = df.groupby(['R_等级', 'M_等级', '裂变类型'])['是否转化'].mean().reset_index()
rf_fission_conversion.rename(columns={'是否转化': '转化率'}, inplace=True)
# 为了简化展示，我们可以先看 R_等级 x 裂变类型
r_fission_conversion = df.groupby(['R_等级', '裂变类型'])['是否转化'].mean().reset_index()
r_fission_conversion.rename(columns={'是否转化': '转化率'}, inplace=True)
analysis_results['r_fission_conversion'] = r_fission_conversion
print("\nR_等级 x 裂变类型的转化率:")
print(r_fission_conversion)

# 历史行为 x 裂变类型 (以 曾助力 为例)
history_fission_conversion = df.groupby(['曾助力', '裂变类型'])['是否转化'].mean().reset_index()
history_fission_conversion.rename(columns={'是否转化': '转化率'}, inplace=True)
analysis_results['history_fission_conversion'] = history_fission_conversion
print("\n是否曾助力 x 裂变类型的转化率:")
print(history_fission_conversion)

print("\n--- 用户细分与个性化策略分析完成 ---")

# 保存部分关键结果到CSV文件
for key, result_df in analysis_results.items():
    result_df.to_csv(f'E:/datalogic-main/10 增长模型/02_用户细分与个性化策略/{key}.csv', index=False, encoding='utf-8-sig')

print("\n关键分析结果已保存为 CSV 文件。")