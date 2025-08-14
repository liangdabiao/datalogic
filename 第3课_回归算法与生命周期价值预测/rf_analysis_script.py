import numpy as np # 导入NumPy
import pandas as pd # 导入Pandas

# --- 数据加载与预处理 (复制自原 notebook) ---
df_sales = pd.read_csv('电商历史订单.csv') # 导入数据集
df_sales['总价'] = df_sales['数量'] * df_sales['单价'] # 计算每单的总价
df_sales['消费日期'] = pd.to_datetime(df_sales['消费日期']) # 转换日期格式

# 构建仅含前3 个月数据的数据集 (特征工程周期)
df_sales_3m = df_sales[(df_sales.消费日期 > '2022-06-01') & (df_sales.消费日期 <= '2022-08-30')]
df_sales_3m.reset_index(drop=True) # 重置索引

# 生成以"用户码"为主键的对象
df_user_LTV = pd.DataFrame(df_sales_3m['用户码'].unique())
df_user_LTV.columns = ['用户码'] # 设定字段名

# 计算 R 值 (最近一次消费距期末天数)
df_R_value = df_sales_3m.groupby('用户码')['消费日期'].max().reset_index()
df_R_value.columns = ['用户码','最近购买日期']
df_R_value['R值'] = (df_R_value['最近购买日期'].max() - df_R_value['最近购买日期']).dt.days
df_user_LTV = pd.merge(df_user_LTV, df_R_value[['用户码','R值']], on='用户码')

# 计算 F 值 (消费次数)
df_F_value = df_sales_3m.groupby('用户码')['消费日期'].count().reset_index()
df_F_value.columns = ['用户码','F值']
df_user_LTV = pd.merge(df_user_LTV, df_F_value[['用户码','F值']], on='用户码')

# 计算 M 值 (总消费金额)
df_M_value = df_sales_3m.groupby('用户码')['总价'].sum().reset_index()
df_M_value.columns = ['用户码','M值']
df_user_LTV = pd.merge(df_user_LTV, df_M_value, on='用户码')

# 计算年度 LTV (目标变量)
df_user_1y = df_sales.groupby('用户码')['总价'].sum().reset_index()
df_user_1y.columns = ['用户码','年度LTV']

# 合并特征和标签，形成最终数据集
df_LTV = pd.merge(df_user_LTV, df_user_1y, on='用户码', how='left')

# --- 准备 Scikit-learn 格式的数据 ---
X = df_LTV.drop(['用户码','年度LTV'],axis=1) # 特征集
y = df_LTV['年度LTV'] # 标签集

# 数据集划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# --- 使用随机森林进行训练和预测 ---
print("--- 使用随机森林 (Random Forest) 进行预测 ---")
from sklearn.ensemble import RandomForestRegressor

# 1. 初始化模型 (使用默认参数进行初步测试)
rf_model = RandomForestRegressor(random_state=42) # 设置 random_state 以确保结果可复现

# 2. 训练模型
rf_model.fit(X_train, y_train)

# 3. 进行预测
y_train_preds_rf = rf_model.predict(X_train)
y_test_preds_rf = rf_model.predict(X_test)

# --- 模型评估 ---
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. 计算 R2 分数
train_r2_rf = r2_score(y_true=y_train, y_pred=y_train_preds_rf)
test_r2_rf = r2_score(y_true=y_test, y_pred=y_test_preds_rf)

print(f"随机森林 - 训练集 R2 分数: {train_r2_rf:.4f}")
print(f"随机森林 - 测试集 R2 分数: {test_r2_rf:.4f}")

# 2. 与原线性回归模型比较
# 原线性回归结果 (从原 notebook 中获取)
lr_test_r2 = 0.4778
print(f"\n--- 模型性能比较 ---")
print(f"线性回归 - 测试集 R2 分数: {lr_test_r2:.4f}")
print(f"随机森林 - 测试集 R2 分数: {test_r2_rf:.4f}")
improvement = test_r2_rf - lr_test_r2
print(f"性能提升 (R2): {improvement:.4f}")

# 3. 可视化预测效果 (测试集)
plt.rcParams["font.family"] = ['SimHei'] # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_preds_rf, alpha=0.5, label='随机森林预测值')
# 添加 y=x 参考线
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='完美预测线')
plt.xlabel('实际 LTV')
plt.ylabel('预测 LTV')
plt.title('随机森林: 实际值 vs. 预测值')
plt.legend()
plt.show()

# --- 特征重要性分析 ---
print("\n--- 特征重要性 ---")
# 获取特征重要性
importances = rf_model.feature_importances_
feature_names = X.columns
# 创建并排序 Series
feature_importance_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(8, 6))
feature_importance_df.plot(kind='barh')
plt.xlabel('Importance Score')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis() # 使最重要的特征在最上方
plt.show()