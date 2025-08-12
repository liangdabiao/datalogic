import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
print("--- 步骤 1: 数据加载与预处理 ---")
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')
print(f"原始数据 shape: {df.shape}")
print(df.head())

# 特征工程：One-Hot Encoding
print("\n进行 One-Hot Encoding...")
df_dummies = pd.get_dummies(df, columns=['设备', '城市类型'])
print(f"One-Hot Encoding 后数据 shape: {df_dummies.shape}")

# 2. 构造 Uplift 标签
print("\n--- 步骤 2: 构造 Uplift 标签 ---")

# 为 '助力砍价' 和 '无裂变页面' 构造标签 (Discount Campaign)
df_discount = df_dummies.query("裂变类型 == '助力砍价' or 裂变类型 == '无裂变页面'").copy()
df_discount.loc[(df_discount['裂变类型'] == '助力砍价') & (df_discount['是否转化'] == 1), '标签'] = 0  # TR
df_discount.loc[(df_discount['裂变类型'] == '助力砍价') & (df_discount['是否转化'] == 0), '标签'] = 1  # TN
df_discount.loc[(df_discount['裂变类型'] == '无裂变页面') & (df_discount['是否转化'] == 1), '标签'] = 2  # CR
df_discount.loc[(df_discount['裂变类型'] == '无裂变页面') & (df_discount['是否转化'] == 0), '标签'] = 3  # CN
df_discount['标签'] = df_discount['标签'].astype(int)
print(f"'助力砍价' vs '无裂变页面' 数据 shape: {df_discount.shape}")
print("'助力砍价' vs '无裂变页面' 标签分布:")
print(df_discount['标签'].value_counts().sort_index())

# 为 '拼团狂买' 和 '无裂变页面' 构造标签 (GroupBuy Campaign)
df_groupbuy = df_dummies.query("裂变类型 == '拼团狂买' or 裂变类型 == '无裂变页面'").copy()
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '拼团狂买') & (df_groupbuy['是否转化'] == 1), '标签'] = 0  # TR
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '拼团狂买') & (df_groupbuy['是否转化'] == 0), '标签'] = 1  # TN
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '无裂变页面') & (df_groupbuy['是否转化'] == 1), '标签'] = 2  # CR
df_groupbuy.loc[(df_groupbuy['裂变类型'] == '无裂变页面') & (df_groupbuy['是否转化'] == 0), '标签'] = 3  # CN
df_groupbuy['标签'] = df_groupbuy['标签'].astype(int)
print(f"\n'拼团狂买' vs '无裂变页面' 数据 shape: {df_groupbuy.shape}")
print("'拼团狂买' vs '无裂变页面' 标签分布:")
print(df_groupbuy['标签'].value_counts().sort_index())

# 3. 准备训练数据
print("\n--- 步骤 3: 准备训练数据 ---")

def prepare_data(df_campaign, campaign_name):
    """为特定活动准备训练/测试数据"""
    # 特征 (X) 和 标签 (y)
    feature_cols = [col for col in df_campaign.columns if col not in ['用户码', '裂变类型', '是否转化', '标签']]
    X = df_campaign[feature_cols]
    y = df_campaign['标签']
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16, stratify=y)
    
    print(f"\n{campaign_name} 数据集:")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"  y_train 标签分布:\n{pd.Series(y_train).value_counts().sort_index()}")
    print(f"  y_test 标签分布:\n{pd.Series(y_test).value_counts().sort_index()}")
    
    return X_train, X_test, y_train, y_test, df_campaign.loc[X_test.index]

# 准备两个活动的数据
X_train_d, X_test_d, y_train_d, y_test_d, df_test_d = prepare_data(df_discount, "助力砍价")
X_train_g, X_test_g, y_train_g, y_test_g, df_test_g = prepare_data(df_groupbuy, "拼团狂买")

print("\n--- 数据准备完成 ---")