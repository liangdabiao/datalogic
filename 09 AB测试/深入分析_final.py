import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 1. 数据加载与初步探索 ---
df = pd.read_csv('AB测试.csv')
print('数据集总用户数:', len(df))
print()
print('数据集前5行:')
print(df.head())
print()
print('数据集基本信息:')
print(df.info())
print()
print('数值型列的描述性统计:')
print(df.describe())

# 确保 '是否购买' 列是字符串类型，以防万一
df['是否购买'] = df['是否购买'].astype(str)

# --- 2. 核心分析：转化率与留存率 ---
# 2.1 整体转化率
print()
print('--- 整体转化率 ---')
conversion_rates = df.groupby('页面版本')['是否购买'].value_counts(normalize=True).unstack(fill_value=0)
print(conversion_rates)
# 可视化整体转化率 - 修正列名引用
# 假设 '是' 和 '否' 是 '是否购买' 列的唯一值
yes_label = '是'
no_label = '否'
# 如果上面的假设不成立，可以动态获取：
# yes_label = df['是否购买'].unique()[df['是否购买'].unique() != '否'][0] 
# no_label = '否'

ax1 = conversion_rates[yes_label].plot(kind='bar', title='整体购买转化率 (按页面版本)', ylabel='转化率', xlabel='页面版本')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('整体转化率.png')
plt.show()

# 2.2 整体留存率
print()
print('--- 整体留存率 ---')
retention_rates = df.groupby('页面版本')['retention_7'].value_counts(normalize=True).unstack(fill_value=0)
print(retention_rates)
# 可视化整体留存率
# retention_7 列是布尔型，unstack后列名是 True 和 False
retention_plot_data = retention_rates[True] if True in retention_rates.columns else pd.Series([0, 0], index=retention_rates.index)
ax2 = retention_plot_data.plot(kind='bar', title='整体7日留存率 (按页面版本)', ylabel='留存率', xlabel='页面版本')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('整体留存率.png')
plt.show()

# --- 3. 用户特征分析 ---
# 3.1 性别分布
print()
print('--- 性别分布 ---')
print(df['性别'].value_counts())
# 3.2 价值组别分布
print()
print('--- 价值组别分布 ---')
print(df['价值组别'].value_counts())

# --- 4. A/B测试细分市场再分析 ---
print()
print('--- A/B测试细分市场再分析 ---')
# 4.1 按价值组别细分的转化率
df_low_value = df[df['价值组别'] == '低']
df_high_value = df[df['价值组别'] == '高']

low_conversion_rates = df_low_value.groupby('页面版本')['是否购买'].value_counts(normalize=True).unstack(fill_value=0)
high_conversion_rates = df_high_value.groupby('页面版本')['是否购买'].value_counts(normalize=True).unstack(fill_value=0)

print()
print('低价值用户转化率:')
print(low_conversion_rates)
print()
print('高价值用户转化率:')
print(high_conversion_rates)

# 可视化细分转化率 - 修正列名引用
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 检查列是否存在，避免KeyError
low_plot_data = low_conversion_rates[yes_label] if yes_label in low_conversion_rates.columns else pd.Series([0, 0], index=low_conversion_rates.index)
high_plot_data = high_conversion_rates[yes_label] if yes_label in high_conversion_rates.columns else pd.Series([0, 0], index=high_conversion_rates.index)

low_plot_data.plot(kind='bar', ax=axes[0], title='低价值用户购买转化率', ylabel='转化率')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(axis='y')

high_plot_data.plot(kind='bar', ax=axes[1], title='高价值用户购买转化率', ylabel='转化率')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].grid(axis='y')
plt.tight_layout()
plt.savefig('细分市场转化率.png')
plt.show()


# 4.2 细分市场的统计检验
# 准备数据进行检验 (在分析之前进行映射)
df['是否购买_数值'] = df['是否购买'].map({'是': 1, '否': 0})

df_test_low = df.loc[(df['页面版本'] == '新页面') & (df['价值组别'] == '低'), '是否购买_数值']
df_control_low = df.loc[(df['页面版本'] == '旧页面') & (df['价值组别'] == '低'), '是否购买_数值']

df_test_high = df.loc[(df['页面版本'] == '新页面') & (df['价值组别'] == '高'), '是否购买_数值']
df_control_high = df.loc[(df['页面版本'] == '旧页面') & (df['价值组别'] == '高'), '是否购买_数值']

def eval_AB_test(test_group, control_group, group_name):
    """执行t检验并打印结果"""
    # 确保输入是数值Series
    test_group = pd.Series(test_group)
    control_group = pd.Series(control_group)
    
    # 执行t检验
    AB_test_result = stats.ttest_ind(test_group, control_group)
    print(f'
{group_name} 用户群体 A/B 测试结果:')
    print('P值：', AB_test_result.pvalue)
    print('t值：', AB_test_result.statistic)
    if AB_test_result.pvalue < 0.05:
        print('结果可信 (p < 0.05)')
    else:
        print('结果不可信 (p >= 0.05)')

eval_AB_test(df_test_low, df_control_low, '低价值')
eval_AB_test(df_test_high, df_control_high, '高价值')

# --- 5. 交互效应分析 ---
print()
print('--- 交互效应分析 ---')
# 5.1 页面版本 * 价值组别 vs. 购买率
interaction_purchase = df.groupby(['页面版本', '价值组别'])['是否购买_数值'].mean().unstack('价值组别')
print()
print('页面版本 * 价值组别 交互效应 (购买率):')
print(interaction_purchase)
# 可视化交互效应
ax3 = interaction_purchase.plot(kind='bar', title='页面版本 vs. 价值组别 交互效应 (购买率)', ylabel='购买率')
plt.xticks(rotation=0)
plt.legend(title='价值组别')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('交互效应_价值组别.png')
plt.show()

# 5.2 页面版本 * 性别 vs. 购买率
interaction_gender_purchase = df.groupby(['页面版本', '性别'])['是否购买_数值'].mean().unstack('性别')
print()
print('页面版本 * 性别 交互效应 (购买率):')
print(interaction_gender_purchase)
# 可视化交互效应
ax4 = interaction_gender_purchase.plot(kind='bar', title='页面版本 vs. 性别 交互效应 (购买率)', ylabel='购买率')
plt.xticks(rotation=0)
plt.legend(title='性别')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('交互效应_性别.png')
plt.show()

# --- 6. 用户分群探索 (基于累计消费次数) ---
print()
print('--- 用户分群探索 (基于累计消费次数) ---')
# 将累计消费次数分为三箱: 低, 中, 高
df['消费分箱'] = pd.qcut(df['累计消费次数'], q=3, labels=['低', '中', '高'])
print()
print('消费分箱分布:')
print(df['消费分箱'].value_counts())

# 分析不同消费水平用户的转化率
consumption_purchase = df.groupby(['页面版本', '消费分箱'])['是否购买_数值'].mean().unstack('消费分箱')
print()
print('页面版本 * 消费分箱 交互效应 (购买率):')
print(consumption_purchase)
# 可视化
ax5 = consumption_purchase.plot(kind='bar', title='页面版本 vs. 消费分箱 交互效应 (购买率)', ylabel='购买率')
plt.xticks(rotation=0)
plt.legend(title='消费分箱')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('交互效应_消费分箱.png')
plt.show()

# --- 7. 稳健性检查：随机分配检验 ---
print()
print('--- 稳健性检查：随机分配检验 ---')
# 检查两组在性别上的分布是否一致
print()
print('旧页面性别分布:')
print(df[df['页面版本'] == '旧页面']['性别'].value_counts(normalize=True))
print()
print('新页面性别分布:')
print(df[df['页面版本'] == '新页面']['性别'].value_counts(normalize=True))
# 卡方检验
gender_crosstab = pd.crosstab(df['页面版本'], df['性别'])
chi2_gender, p_gender, dof_gender, expected_gender = stats.chi2_contingency(gender_crosstab)
print(f'
性别分布卡方检验: chi2={chi2_gender:.4f}, p={p_gender:.4f}')
if p_gender > 0.05:
    print('两组性别分布无显著差异，随机分配假设成立。')
else:
    print('两组性别分布存在显著差异，可能存在选择偏差。')

# 检查两组在价值组别上的分布是否一致
print()
print('旧页面价值组别分布:')
print(df[df['页面版本'] == '旧页面']['价值组别'].value_counts(normalize=True))
print()
print('新页面价值组别分布:')
print(df[df['页面版本'] == '新页面']['价值组别'].value_counts(normalize=True))
# 卡方检验
value_crosstab = pd.crosstab(df['页面版本'], df['价值组别'])
chi2_value, p_value, dof_value, expected_value = stats.chi2_contingency(value_crosstab)
print(f'
价值组别分布卡方检验: chi2={chi2_value:.4f}, p={p_value:.4f}')
if p_value > 0.05:
    print('两组价值组别分布无显著差异，随机分配假设成立。')
else:
    print('两组价值组别分布存在显著差异，可能存在选择偏差。')

# 为了方便查看，将关键的交叉表结果也保存到CSV
interaction_purchase.to_csv('交互效应_价值组别_结果.csv')
interaction_gender_purchase.to_csv('交互效应_性别_结果.csv')
consumption_purchase.to_csv('交互效应_消费分箱_结果.csv')
print("
关键的交互效应分析结果已保存到 CSV 文件。")
