#!/usr/bin/env python
# coding: utf-8

# # 乳腺检查数据 Logistic Regression 建模

# ## 1. 导入必要的库

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# ## 2. 数据加载与初步探索

# In[2]:


# 尝试不同的编码方式加载数据
# data = pd.read_csv('某地乳腺检查数据.csv', encoding='utf-8') # 可能会因BOM出错
# data = pd.read_csv('某地乳腺检查数据.csv', encoding='utf-8-sig') # 推荐
# data = pd.read_csv('某地乳腺检查数据.csv', encoding='ISO-8859-1') # 如果上述方式不行可尝试

# 为了解决中文乱码问题，我们先用默认方式读取，然后手动处理列名
data = pd.read_csv('某地乳腺检查数据.csv')
# 打印原始列名以确认
print("原始列名 (可能乱码):")
print(data.columns.tolist())
print("\n原始数据前几行:")
print(data.head())


# In[3]:


# 手动指定正确的列名 (基于对数据结构的理解)
# 假设列的顺序是正确的，我们重新命名
correct_columns = [
    'ID', '诊断结果', '平均半径', '平均纹理', '平均周长', '平均面积', '平均光滑度', '平均致密度', '平均凹度', '平均凹点数', '平均对称性', '平均分形维度',
    '半径的标准误差', '纹理的标准误差', '周长的标准误差', '面积的标准误差', '光滑度的标准误差', '致密度的标准误差', '凹度的标准误差', '凹点数的标准误差', '对称性的标准误差', '分形维度的标准误差',
    '最差半径', '最差纹理', '最差周长', '最差面积', '最差光滑度', '最差致密度', '最差凹度', '最差凹点数', '最差对称性', '最差分形维度'
]
# 检查列表长度是否匹配
if len(correct_columns) == data.shape[1]:
    data.columns = correct_columns
    print("\n列名已成功修正。")
else:
    print("\n警告：手动指定的列名数量 ({}) 与数据列数 ({}) 不匹配。请检查。".format(len(correct_columns), data.shape[1]))
    # 如果不匹配，我们仍然尝试用原始列名进行操作，但需要特别注意


# In[4]:


# 检查数据形状
print("\n数据集形状:", data.shape)


# In[5]:


# 查看前几行数据
print("\n修正列名后的数据:")
data.head()


# In[6]:


# 查看修正后的列名
print("\n修正后的列名:")
print(data.columns.tolist())


# In[7]:


# 检查目标变量 '诊断结果' 的分布
print("\n诊断结果分布:")
print(data['诊断结果'].value_counts())


# In[8]:


# 检查数据类型
print("\n数据类型:")
print(data.dtypes)


# In[9]:


# 检查是否有缺失值
print("\n缺失值检查:")
print(data.isnull().sum())


# ## 3. 数据预处理

# In[10]:


# 处理目标变量 '诊断结果'
# 使用 LabelEncoder 将其转换为数值 (确诊=1, 健康=0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['诊断结果'])

# 查看编码后的结果
print("\n目标变量编码:")
print("原始标签:", data['诊断结果'].unique())
print("编码后标签:", label_encoder.classes_)
print("编码映射:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# In[11]:


# 选择特征变量 (移除 'ID' 和 '诊断结果')
feature_columns = [col for col in data.columns if col not in ['ID', '诊断结果']]
X = data[feature_columns]

# 检查特征矩阵形状
print("\n特征矩阵形状:", X.shape)


# In[12]:


# 划分训练集和测试集 (70% 训练, 30% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\n数据集划分完成:")
print("训练集特征形状:", X_train.shape)
print("训练集标签形状:", y_train.shape)
print("测试集特征形状:", X_test.shape)
print("测试集标签形状:", y_test.shape)


# In[13]:


# 特征缩放 (标准化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换回 DataFrame 以便查看 (可选)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_columns)

print("\n特征缩放完成。标准化后的训练集特征 (前5行):")
X_train_scaled_df.head()


# ## 4. 模型构建与训练

# In[14]:


# 实例化逻辑回归模型
# 增加 max_iter 以确保收敛
model = LogisticRegression(max_iter=1000, random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)
print("\n模型训练完成。")


# ## 5. 模型评估

# In[15]:


# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # 获取属于类别1(确诊)的概率

# 计算评估指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n模型在测试集上的性能:")
print("准确率 (Accuracy): {:.4f}".format(acc))
print("精确率 (Precision): {:.4f}".format(prec))
print("召回率 (Recall): {:.4f}".format(rec))
print("F1分数 (F1-Score): {:.4f}".format(f1))
print("AUC (Area Under ROC Curve): {:.4f}".format(auc))


# In[16]:


# 设置中文字体以解决图表乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# In[17]:


# 绘制混淆矩阵并保存
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n混淆矩阵已保存为 'confusion_matrix.png'")


# In[18]:


# 绘制ROC曲线并保存
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nROC曲线已保存为 'roc_curve.png'")


# ## 6. 结果解释 (可选)

# In[19]:


# 查看模型系数
coefficients = pd.DataFrame(model.coef_[0], index=feature_columns, columns=['Coefficient'])
coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
coefficients_sorted = coefficients.sort_values(by='Abs_Coefficient', ascending=False)

print("\n模型系数 (按绝对值大小排序，前10个影响力最大的特征):")
print(coefficients_sorted.head(10))


# ## 7. 总结与反思

# In[20]:


print("\n\n总结与反思:")
print("此逻辑回归模型在乳腺检查数据上表现良好，各项指标均较高。")
print("\n优点:")
print("1. 模型简单、可解释性强。")
print("2. 训练速度快。")
print("3. 在此数据集上取得了不错的性能。")
print("\n可能的改进方向:")
print("1. 特征工程: 探索特征组合、多项式特征等。")
print("2. 超参数调优: 使用网格搜索 (GridSearchCV) 或随机搜索 (RandomizedSearchCV) 来优化 C (正则化强度) 和 penalty (正则化类型) 等参数。")
print("3. 尝试其他模型: 如支持向量机 (SVM), 随机森林 (Random Forest), 梯度提升树 (XGBoost, LightGBM) 等，比较性能。")
print("4. 交叉验证: 使用 cross_val_score 进行更稳健的性能评估。")