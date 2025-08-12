# 注意：这个脚本需要与 data_preprocessing.py 在同一个会话中运行，或者需要加载预处理后的数据。
# 为了简化，我们将两个步骤合并到一个脚本中执行。

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- 数据预处理部分 (来自 data_preprocessing.py) ---
print("--- 数据预处理 ---")
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')
df_dummies = pd.get_dummies(df, columns=['设备', '城市类型'])

# 构造 Uplift 标签
print("\n--- 构造 Uplift 标签 ---")
df_discount = df_dummies.query("裂变类型 == '助力砍价' or 裂变类型 == '无裂变页面'").copy()
df_discount.loc[(df_discount['裂变类型'] == '助力砍价') & (df_discount['是否转化'] == 1), '标签'] = 0  # TR
df_discount.loc[(df_discount['裂变类型'] == '助力砍价') & (df_discount['是否转化'] == 0), '标签'] = 1  # TN
df_discount.loc[(df_discount['裂变类型'] == '无裂变页面') & (df_discount['是否转化'] == 1), '标签'] = 2  # CR
df_discount.loc[(df_discount['裂变类型'] == '无裂变页面') & (df_discount['是否转化'] == 0), '标签'] = 3  # CN
df_discount['标签'] = df_discount['标签'].astype(int)

df_groupbuy = df_dummies.query("裂变类型 == '拼团狂买' or 裂变类型 == '无裂变页面'").copy()
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '拼团狂买') & (df_groupbuy['是否转化'] == 1), '标签'] = 0  # TR
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '拼团狂买') & (df_groupbuy['是否转化'] == 0), '标签'] = 1  # TN
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '无裂变页面') & (df_groupbuy['是否转化'] == 1), '标签'] = 2  # CR
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '无裂变页面') & (df_groupbuy['是否转化'] == 0), '标签'] = 3  # CN
df_groupbuy['标签'] = df_groupbuy['标签'].astype(int)

# 准备训练数据
def prepare_data(df_campaign):
    feature_cols = [col for col in df_campaign.columns if col not in ['用户码', '裂变类型', '是否转化', '标签']]
    X = df_campaign[feature_cols]
    y = df_campaign['标签']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16, stratify=y)
    return X_train, X_test, y_train, y_test, df_campaign.loc[X_test.index]

X_train_d, X_test_d, y_train_d, y_test_d, df_test_d = prepare_data(df_discount)
X_train_g, X_test_g, y_train_g, y_test_g, df_test_g = prepare_data(df_groupbuy)

print("\n--- 数据预处理完成 ---")

# --- XGBoost Uplift 建模部分 ---
print("\n--- XGBoost Uplift 建模 ---")

# 1. 训练模型
print("\n1. 训练模型...")
xgb_model_d = xgb.XGBClassifier(objective='multi:softprob', num_class=4, random_state=16, n_estimators=100)
xgb_model_d.fit(X_train_d, y_train_d)
print("  '助力砍价' 模型训练完成。")

xgb_model_g = xgb.XGBClassifier(objective='multi:softprob', num_class=4, random_state=16, n_estimators=100)
xgb_model_g.fit(X_train_g, y_train_g)
print("  '拼团狂买' 模型训练完成。")

# 2. 模型预测与增量分数计算
print("\n2. 计算增量分数...")
def calculate_uplift_scores(model, X_test, df_test, campaign_name):
    pred_probs = model.predict_proba(X_test)
    P_TR, P_TN, P_CR, P_CN = pred_probs[:, 0], pred_probs[:, 1], pred_probs[:, 2], pred_probs[:, 3]
    results = df_test[['用户码', '裂变类型', '是否转化']].copy()
    results['P_TR'], results['P_TN'], results['P_CR'], results['P_CN'] = P_TR, P_TN, P_CR, P_CN
    epsilon = 1e-15
    uplift_score = (P_TR - P_TN) / (P_TR + P_TN + epsilon) + (P_CN - P_CR) / (P_CN + P_CR + epsilon)
    results['增量分数'] = uplift_score
    print(f"  '{campaign_name}' 平均增量分数: {uplift_score.mean():.4f}")
    return results

uplift_results_d = calculate_uplift_scores(xgb_model_d, X_test_d, df_test_d, "助力砍价")
uplift_results_g = calculate_uplift_scores(xgb_model_g, X_test_g, df_test_g, "拼团狂买")

# 3. 保存模型和结果
print("\n3. 保存模型和结果...")
joblib.dump(xgb_model_d, 'E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/xgb_model_discount.pkl')
joblib.dump(xgb_model_g, 'E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/xgb_model_groupbuy.pkl')

uplift_results_d.to_csv('E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/uplift_scores_discount.csv', index=False, encoding='utf-8-sig')
uplift_results_g.to_csv('E:/datalogic-main/10 增长模型/05_增长建模_xgboost_qini/uplift_scores_groupbuy.csv', index=False, encoding='utf-8-sig')

print("  模型和增量分数已保存。")

print("\n--- XGBoost Uplift 建模完成 ---")