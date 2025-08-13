import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. 数据加载
user_behavior_path = 'E:/datalogic-main/08 推荐系统/用户行为数据.csv'
product_info_path = 'E:/datalogic-main/08 推荐系统/商品信息数据.csv'

df_user_behavior = pd.read_csv(user_behavior_path, encoding='utf-8')
df_product_info = pd.read_csv(product_info_path, encoding='utf-8')

print("--- 用户行为数据 ---")
print(df_user_behavior.head())
print("\n--- 商品信息数据 ---")
print(df_product_info.head())

# 2. 数据清洗与预处理
# 检查缺失值
print("\n--- 用户行为数据缺失值 ---")
print(df_user_behavior.isnull().sum())
print("\n--- 商品信息数据缺失值 ---")
print(df_product_info.isnull().sum())

# 检查数据类型
print("\n--- 用户行为数据类型 ---")
print(df_user_behavior.dtypes)
print("\n--- 商品信息数据类型 ---")
print(df_product_info.dtypes)

# 确保评分是数值型
df_user_behavior['评分'] = pd.to_numeric(df_user_behavior['评分'], errors='coerce')
# 确保价格是数值型
df_user_behavior['商品价格'] = pd.to_numeric(df_user_behavior['商品价格'], errors='coerce')
df_product_info['价格'] = pd.to_numeric(df_product_info['价格'], errors='coerce')

# 处理可能的缺失评分（用平均分填充，或删除）
# 这里我们选择删除评分为空的行
df_user_behavior.dropna(subset=['评分'], inplace=True)

# 3. 特征工程
# a. 计算商品热度 (销量、平均评分、被浏览/购买的总次数)
# 商品信息数据中已有销量和平均评分，我们可以结合行为数据中的“浏览”和“购买”次数
# 这里简化处理，直接使用商品信息数据中的销量和平均评分作为热度指标
# 如果需要更精确，可以结合行为数据重新计算

# b. 构建用户-商品评分矩阵 (用于协同过滤)
# pivot_table: 行是用户ID，列是商品ID，值是评分
user_item_matrix = df_user_behavior.pivot_table(index='用户ID', columns='商品ID', values='评分', fill_value=0)
print("\n--- 用户-商品评分矩阵 (前5行5列) ---")
print(user_item_matrix.iloc[:5, :5])

# 4. 简单分析
# a. 最受欢迎的商品 (按平均评分和销量)
top_rated_products = df_product_info.sort_values(by='平均评分', ascending=False).head(10)
print("\n--- 平均评分最高的商品 (前10) ---")
print(top_rated_products[['商品ID', '商品名称', '平均评分']])

top_sold_products = df_product_info.sort_values(by='销量', ascending=False).head(10)
print("\n--- 销量最高的商品 (前10) ---")
print(top_sold_products[['商品ID', '商品名称', '销量']])

# b. 用户行为分析
# 平均评分分布
print("\n--- 用户评分分布 ---")
print(df_user_behavior['评分'].value_counts().sort_index())

# 用户购买/浏览次数统计
user_activity = df_user_behavior.groupby('用户ID')['行为类型'].count().sort_values(ascending=False)
print("\n--- 最活跃用户 (行为次数) ---")
print(user_activity.head(10))

# 商品被评分次数统计
item_popularity = df_user_behavior['商品ID'].value_counts().sort_values(ascending=False)
print("\n--- 最热门商品 (被评分/交互次数) ---")
print(item_popularity.head(10))

# 5. 准备协同过滤模型 (基于用户的协同过滤)
# 计算用户相似度矩阵
# 使用余弦相似度
# 注意：如果数据量很大，直接计算用户相似度矩阵会非常消耗内存
# 这里我们先用一个小子集演示
# 选择评分矩阵中的前100个用户进行演示
sample_users = user_item_matrix.index[:100]
user_item_matrix_sample = user_item_matrix.loc[sample_users]

# 标准化评分矩阵 (可选，有时能提高相似度计算的准确性)
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix_sample.T).T # 转置后标准化用户向量
user_item_matrix_scaled = pd.DataFrame(user_item_matrix_scaled, index=user_item_matrix_sample.index, columns=user_item_matrix_sample.columns)

# 计算用户相似度 (使用标准化后的数据)
user_similarity_matrix = cosine_similarity(user_item_matrix_scaled)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix_sample.index, columns=user_item_matrix_sample.index)

print("\n--- 用户相似度矩阵 (前5x5) ---")
print(user_similarity_df.iloc[:5, :5])

# 6. 简单的基于用户协同过滤推荐示例
def recommend_items_for_user(user_id, user_item_matrix, user_similarity_df, n_recommendations=5):
    """
    为指定用户推荐商品
    :param user_id: 目标用户ID
    :param user_item_matrix: 用户-商品评分矩阵
    :param user_similarity_df: 用户相似度矩阵
    :param n_recommendations: 推荐商品数量
    :return: 推荐商品列表
    """
    if user_id not in user_item_matrix.index:
        print(f"用户 {user_id} 不在数据集中。")
        return []

    # 获取与目标用户最相似的用户（排除自己）
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:] # [1:] 排除自己
    
    # 获取目标用户已评分的商品
    user_rated_items = user_item_matrix.loc[user_id]
    user_rated_item_ids = set(user_rated_items[user_rated_items > 0].index.tolist())

    # 收集相似用户喜欢的商品
    recommendations = {}
    for similar_user_id, similarity_score in similar_users.items():
        # 获取相似用户评分的商品
        similar_user_ratings = user_item_matrix.loc[similar_user_id]
        for item_id, rating in similar_user_ratings.items():
            # 只推荐目标用户未评分的商品
            if item_id not in user_rated_item_ids and rating > 0:
                if item_id in recommendations:
                    recommendations[item_id] += similarity_score * rating
                else:
                    recommendations[item_id] = similarity_score * rating
    
    # 按加权评分排序并返回Top N
    recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # 获取商品名称
    recommended_item_ids = [item[0] for item in recommended_items]
    recommended_item_names = df_product_info[df_product_info['商品ID'].isin(recommended_item_ids)][['商品ID', '商品名称']]
    
    return recommended_item_names

# 为用户 U001 推荐商品
target_user = 'U001'
recommended_items = recommend_items_for_user(target_user, user_item_matrix_sample, user_similarity_df, n_recommendations=5)
print(f"\n--- 为用户 {target_user} 推荐的商品 ---")
print(recommended_items)