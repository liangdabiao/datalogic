# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
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

# 确保 '是否购买' 列是字符串类型
df['是否购买'] = df['是否购买'].astype(str)

# --- 2. 核心分析：转化率与留存率 ---
# 2.1 整体转化率
print()
print('--- 整体转化率 ---')
conversion_rates = df.groupby('页面版本')['是否购买'].value_counts(normalize=True).unstack(fill_value=0)
print(conversion_rates)

# 明确使用 '是' 作为列名来获取转化率
yes_label = '是'
no_label = '否'

# 绘图并保存
if yes_label in conversion_rates.columns:
    ax1 = conversion_rates[yes_label].plot(kind='bar', title='整体购买转化率 (按页面版本)', ylabel='转化率', xlabel='页面版本')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('整体转化率.png')
    plt.close() # 关闭图形，释放内存
    print("图表 '整体转化率.png' 已保存。")
else:
    print(f"警告：未找到 '{yes_label}' 列用于绘制转化率图。")

# 2.2 整体留存率
print()
print('--- 整体留存率 ---')
retention_rates = df.groupby('页面版本')['retention_7'].value_counts(normalize=True).unstack(fill_value=0)
print(retention_rates)

# 绘图并保存
if True in retention_rates.columns:
    ax2 = retention_rates[True].plot(kind='bar', title='整体7日留存率 (按页面版本)', ylabel='留存率', xlabel='页面版本')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('整体留存率.png')
    plt.close()
    print("图表 '整体留存率.png' 已保存。")
else:
    print("警告：未找到 True 列用于绘制留存率图。")

# --- 3. 用户特征分析 ---
# 3.1 性别分布
print()
print('--- 性别分布 ---')
gender_counts = df['性别'].value_counts()
print(gender_counts)
# 3.2 价值组别分布
print()
print('--- 价值组别分布 ---')
value_counts = df['价值组别'].value_counts()
print(value_counts)

# --- 4. A/B测试细分市场再分析 ---
print()
print('--- A/B测试细分市场再分析 ---')
# 数据预处理：将 '是否购买' 映射为数值
df['是否购买_数值'] = df['是否购买'].map({'是': 1, '否': 0})

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

# 可视化细分转化率
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 为低价值用户绘图
if yes_label in low_conversion_rates.columns:
    low_conversion_rates[yes_label].plot(kind='bar', ax=axes[0], title='低价值用户购买转化率', ylabel='转化率')
else:
    pd.Series([0, 0], index=low_conversion_rates.index).plot(kind='bar', ax=axes[0], title='低价值用户购买转化率', ylabel='转化率')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(axis='y')

# 为高价值用户绘图
if yes_label in high_conversion_rates.columns:
    high_conversion_rates[yes_label].plot(kind='bar', ax=axes[1], title='高价值用户购买转化率', ylabel='转化率')
else:
    pd.Series([0, 0], index=high_conversion_rates.index).plot(kind='bar', ax=axes[1], title='高价值用户购买转化率', ylabel='转化率')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].grid(axis='y')

plt.tight_layout()
plt.savefig('细分市场转化率.png')
plt.close()
print("图表 '细分市场转化率.png' 已保存。")

# 4.2 细分市场的统计检验
def eval_AB_test(test_group, control_group, group_name):
    """执行t检验并打印结果"""
    test_group = pd.Series(test_group)
    control_group = pd.Series(control_group)
    
    if len(test_group) == 0 or len(control_group) == 0:
        print(f"\n{group_name} 用户群体 A/B 测试结果:")
        print("警告：测试组或对照组为空，无法进行检验。")
        return
        
    AB_test_result = stats.ttest_ind(test_group, control_group)
    print(f"\n{group_name} 用户群体 A/B 测试结果:")
    print('P值：', AB_test_result.pvalue)
    print('t值：', AB_test_result.statistic)
    if AB_test_result.pvalue < 0.05:
        print('结果可信 (p < 0.05)')
    else:
        print('结果不可信 (p >= 0.05)')

# 准备数据
df_test_low = df.loc[(df['页面版本'] == '新页面') & (df['价值组别'] == '低'), '是否购买_数值']
df_control_low = df.loc[(df['页面版本'] == '旧页面') & (df['价值组别'] == '低'), '是否购买_数值']

df_test_high = df.loc[(df['页面版本'] == '新页面') & (df['价值组别'] == '高'), '是否购买_数值']
df_control_high = df.loc[(df['页面版本'] == '旧页面') & (df['价值组别'] == '高'), '是否购买_数值']

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

ax3 = interaction_purchase.plot(kind='bar', title='页面版本 vs. 价值组别 交互效应 (购买率)', ylabel='购买率')
plt.xticks(rotation=0)
plt.legend(title='价值组别')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('交互效应_价值组别.png')
plt.close()
print("图表 '交互效应_价值组别.png' 已保存。")

# 5.2 页面版本 * 性别 vs. 购买率
interaction_gender_purchase = df.groupby(['页面版本', '性别'])['是否购买_数值'].mean().unstack('性别')
print()
print('页面版本 * 性别 交互效应 (购买率):')
print(interaction_gender_purchase)

ax4 = interaction_gender_purchase.plot(kind='bar', title='页面版本 vs. 性别 交互效应 (购买率)', ylabel='购买率')
plt.xticks(rotation=0)
plt.legend(title='性别')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('交互效应_性别.png')
plt.close()
print("图表 '交互效应_性别.png' 已保存。")

# --- 6. 用户分群探索 (基于累计消费次数) ---
print()
print('--- 用户分群探索 (基于累计消费次数) ---')
df['消费分箱'] = pd.qcut(df['累计消费次数'], q=3, labels=['低', '中', '高'])
print()
print('消费分箱分布:')
bin_counts = df['消费分箱'].value_counts()
print(bin_counts)

consumption_purchase = df.groupby(['页面版本', '消费分箱'])['是否购买_数值'].mean().unstack('消费分箱')
print()
print('页面版本 * 消费分箱 交互效应 (购买率):')
print(consumption_purchase)

ax5 = consumption_purchase.plot(kind='bar', title='页面版本 vs. 消费分箱 交互效应 (购买率)', ylabel='购买率')
plt.xticks(rotation=0)
plt.legend(title='消费分箱')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('交互效应_消费分箱.png')
plt.close()
print("图表 '交互效应_消费分箱.png' 已保存。")

# --- 7. 稳健性检查：随机分配检验 ---
print()
print('--- 稳健性检查：随机分配检验 ---')

print()
print('旧页面性别分布:')
old_page_gender = df[df['页面版本'] == '旧页面']['性别'].value_counts(normalize=True)
print(old_page_gender)
print()
print('新页面性别分布:')
new_page_gender = df[df['页面版本'] == '新页面']['性别'].value_counts(normalize=True)
print(new_page_gender)

gender_crosstab = pd.crosstab(df['页面版本'], df['性别'])
chi2_gender, p_gender, dof_gender, expected_gender = stats.chi2_contingency(gender_crosstab)
print(f"\n性别分布卡方检验: chi2={chi2_gender:.4f}, p={p_gender:.4f}")
if p_gender > 0.05:
    conclusion_gender = '两组性别分布无显著差异，随机分配假设成立。'
else:
    conclusion_gender = '两组性别分布存在显著差异，可能存在选择偏差。'
print(conclusion_gender)

print()
print('旧页面价值组别分布:')
old_page_value = df[df['页面版本'] == '旧页面']['价值组别'].value_counts(normalize=True)
print(old_page_value)
print()
print('新页面价值组别分布:')
new_page_value = df[df['页面版本'] == '新页面']['价值组别'].value_counts(normalize=True)
print(new_page_value)

value_crosstab = pd.crosstab(df['页面版本'], df['价值组别'])
chi2_value, p_value, dof_value, expected_value = stats.chi2_contingency(value_crosstab)
print(f"\n价值组别分布卡方检验: chi2={chi2_value:.4f}, p={p_value:.4f}")
if p_value > 0.05:
    conclusion_value = '两组价值组别分布无显著差异，随机分配假设成立。'
else:
    conclusion_value = '两组价值组别分布存在显著差异，可能存在选择偏差。'
print(conclusion_value)

# 保存关键结果到文本文件
with open('分析摘要.txt', 'w', encoding='utf-8') as f:
    f.write("=== AB测试深入分析摘要 ===\n\n")
    
    f.write("--- 核心指标 ---\n")
    f.write(f"整体转化率:\n{conversion_rates.to_string()}\n\n")
    f.write(f"整体留存率:\n{retention_rates.to_string()}\n\n")
    
    f.write("--- 细分市场转化率 ---\n")
    f.write(f"低价值用户转化率:\n{low_conversion_rates.to_string()}\n\n")
    f.write(f"高价值用户转化率:\n{high_conversion_rates.to_string()}\n\n")
    
    f.write("--- 统计检验结果 ---\n")
    # 重新执行检验以捕获输出
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    eval_AB_test(df_test_low, df_control_low, '低价值')
    eval_AB_test(df_test_high, df_control_high, '高价值')
    sys.stdout = old_stdout
    f.write(mystdout.getvalue())
    f.write("\n")
    
    f.write("--- 稳健性检查结论 ---\n")
    f.write(f"性别分布卡方检验: chi2={chi2_gender:.4f}, p={p_gender:.4f}\n")
    f.write(f"{conclusion_gender}\n\n")
    f.write(f"价值组别分布卡方检验: chi2={chi2_value:.4f}, p={p_value:.4f}\n")
    f.write(f"{conclusion_value}\n\n")
    
    f.write("--- 用户特征分布 ---\n")
    f.write(f"性别:\n{gender_counts.to_string()}\n\n")
    f.write(f"价值组别:\n{value_counts.to_string()}\n\n")
    f.write(f"消费分箱:\n{bin_counts.to_string()}\n\n")

print("\n分析摘要已保存到 '分析摘要.txt'。")
print("\n所有分析已完成，图表和结果均已生成。")