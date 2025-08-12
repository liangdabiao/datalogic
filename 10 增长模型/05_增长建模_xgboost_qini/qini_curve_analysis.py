import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def qini_curve(uplift_scores_df, campaign_name):
    """
    计算并绘制 Qini 曲线
    :param uplift_scores_df: 包含 '用户码', '裂变类型', '是否转化', '增量分数' 的 DataFrame
    :param campaign_name: 活动名称，用于标题和文件名
    :return: matplotlib 图形对象
    """
    print(f"\n--- 计算 '{campaign_name}' 的 Qini 曲线 ---")
    
    # 1. 数据排序
    df_sorted = uplift_scores_df.sort_values(by='增量分数', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1
    df_sorted['fraction'] = df_sorted['rank'] / len(df_sorted) # 累积比例

    # 2. 计算 TR (Treatment Responders) 和 CR (Control Responders) 数量
    # TR: 看到裂变页面且转化的用户
    # CR: 看到无裂变页面且转化的用户
    tr_total = len(df_sorted[(df_sorted['裂变类型'] != '无裂变页面') & (df_sorted['是否转化'] == 1)])
    cr_total = len(df_sorted[(df_sorted['裂变类型'] == '无裂变页面') & (df_sorted['是否转化'] == 1)])
    
    print(f"  TR 总数: {tr_total}, CR 总数: {cr_total}")

    # 3. 计算累积增益 (Cumulative Gain)
    df_sorted['is_tr'] = ((df_sorted['裂变类型'] != '无裂变页面') & (df_sorted['是否转化'] == 1)).astype(int)
    df_sorted['is_cr'] = ((df_sorted['裂变类型'] == '无裂变页面') & (df_sorted['是否转化'] == 1)).astype(int)
    
    df_sorted['cum_tr'] = df_sorted['is_tr'].cumsum()
    df_sorted['cum_cr'] = df_sorted['is_cr'].cumsum()
    
    # Qini 增益 = Cumulative TR Rate - Cumulative CR Rate
    #           = (Cum_TR / Total_TR) - (Cum_CR / Total_CR)
    df_sorted['qini_gain'] = (df_sorted['cum_tr'] / tr_total) - (df_sorted['cum_cr'] / cr_total)

    # 4. 计算随机模型的 Qini 增益 (一条直线)
    df_sorted['random_qini_gain'] = df_sorted['fraction'] * df_sorted['qini_gain'].iloc[-1]

    # 5. 绘制 Qini 曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 实际模型曲线
    ax.plot(df_sorted['fraction'], df_sorted['qini_gain'], label=f'{campaign_name} Qini Curve', linewidth=2)
    # 随机模型曲线
    ax.plot(df_sorted['fraction'], df_sorted['random_qini_gain'], label='Random', linestyle='--', linewidth=2)
    
    # 填充两条曲线之间的区域，表示模型的优势
    ax.fill_between(df_sorted['fraction'], df_sorted['qini_gain'], df_sorted['random_qini_gain'], alpha=0.3, label='Model Gain')
    
    ax.set_xlabel('用户累积占比 (%)', fontsize=12)
    ax.set_ylabel('Qini 增益', fontsize=12)
    ax.set_title(f'Qini Curve - {campaign_name}', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加auc
    auc_qini = np.trapz(df_sorted['qini_gain'], df_sorted['fraction'])
    auc_random = np.trapz(df_sorted['random_qini_gain'], df_sorted['fraction'])
    print(f"  '{campaign_name}' Qini AUC: {auc_qini:.4f}")
    print(f"  Random Qini AUC: {auc_random:.4f}")
    print(f"  '{campaign_name}' 相对提升 AUC: {(auc_qini - auc_random):.4f}")

    plt.tight_layout()
    # plt.savefig(f'E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/qini_curve_{campaign_name}.png', dpi=300, bbox_inches='tight')
    
    print(f"  '{campaign_name}' Qini 曲线计算并绘图完成。")
    return fig

# --- 主程序 ---
print("--- 开始 Qini 曲线分析 ---")

# 1. 加载增量分数数据
print("\n1. 加载增量分数数据...")
uplift_scores_d = pd.read_csv('E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/uplift_scores_discount.csv')
uplift_scores_g = pd.read_csv('E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/uplift_scores_groupbuy.csv')

# 2. 绘制 Qini 曲线
print("\n2. 绘制 Qini 曲线...")
fig_d = qini_curve(uplift_scores_d, "助力砍价")
fig_g = qini_curve(uplift_scores_g, "拼团狂买")

# 3. 比较两条 Qini 曲线
print("\n3. 比较两条 Qini 曲线...")
fig_compare, ax = plt.subplots(figsize=(10, 7))

# 重新计算用于比较的数据
# 助力砍价
df_d_sorted = uplift_scores_d.sort_values(by='增量分数', ascending=False).reset_index(drop=True)
df_d_sorted['fraction'] = (df_d_sorted.index + 1) / len(df_d_sorted)
tr_d_total = len(df_d_sorted[(df_d_sorted['裂变类型'] != '无裂变页面') & (df_d_sorted['是否转化'] == 1)])
cr_d_total = len(df_d_sorted[(df_d_sorted['裂变类型'] == '无裂变页面') & (df_d_sorted['是否转化'] == 1)])
df_d_sorted['is_tr'] = ((df_d_sorted['裂变类型'] != '无裂变页面') & (df_d_sorted['是否转化'] == 1)).astype(int)
df_d_sorted['is_cr'] = ((df_d_sorted['裂变类型'] == '无裂变页面') & (df_d_sorted['是否转化'] == 1)).astype(int)
df_d_sorted['cum_tr'] = df_d_sorted['is_tr'].cumsum()
df_d_sorted['cum_cr'] = df_d_sorted['is_cr'].cumsum()
df_d_sorted['qini_gain'] = (df_d_sorted['cum_tr'] / tr_d_total) - (df_d_sorted['cum_cr'] / cr_d_total)

# 拼团狂买
df_g_sorted = uplift_scores_g.sort_values(by='增量分数', ascending=False).reset_index(drop=True)
df_g_sorted['fraction'] = (df_g_sorted.index + 1) / len(df_g_sorted)
tr_g_total = len(df_g_sorted[(df_g_sorted['裂变类型'] != '无裂变页面') & (df_g_sorted['是否转化'] == 1)])
cr_g_total = len(df_g_sorted[(df_g_sorted['裂变类型'] == '无裂变页面') & (df_g_sorted['是否转化'] == 1)])
df_g_sorted['is_tr'] = ((df_g_sorted['裂变类型'] != '无裂变页面') & (df_g_sorted['是否转化'] == 1)).astype(int)
df_g_sorted['is_cr'] = ((df_g_sorted['裂变类型'] == '无裂变页面') & (df_g_sorted['是否转化'] == 1)).astype(int)
df_g_sorted['cum_tr'] = df_g_sorted['is_tr'].cumsum()
df_g_sorted['cum_cr'] = df_g_sorted['is_cr'].cumsum()
df_g_sorted['qini_gain'] = (df_g_sorted['cum_tr'] / tr_g_total) - (df_g_sorted['cum_cr'] / cr_g_total)

# 绘制
ax.plot(df_d_sorted['fraction'], df_d_sorted['qini_gain'], label='助力砍价 Qini Curve', linewidth=2)
ax.plot(df_g_sorted['fraction'], df_g_sorted['qini_gain'], label='拼团狂买 Qini Curve', linewidth=2)
ax.plot([0, 1], [0, max(df_d_sorted['qini_gain'].iloc[-1], df_g_sorted['qini_gain'].iloc[-1])], label='Random', linestyle='--', linewidth=2)

ax.set_xlabel('用户累积占比 (%)', fontsize=12)
ax.set_ylabel('Qini 增益', fontsize=12)
ax.set_title('Qini Curve Comparison - 助力砍价 vs 拼团狂买', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
# plt.savefig('E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/qini_curve_comparison.png', dpi=300, bbox_inches='tight')

plt.show()

print("\n--- Qini 曲线分析完成 ---")