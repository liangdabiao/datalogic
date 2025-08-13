import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 设置 pandas 显示选项，避免科学计数法
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 1. 数据加载
user_behavior_path = 'E:/datalogic-main/08 推荐系统/用户行为数据.csv'
product_info_path = 'E:/datalogic-main/08 推荐系统/商品信息数据.csv'

df_user_behavior = pd.read_csv(user_behavior_path, encoding='utf-8')
df_product_info = pd.read_csv(product_info_path, encoding='utf-8')

print("--- 数据加载完成 ---")

# 2. 数据清洗与预处理
# 确保评分是数值型
df_user_behavior['评分'] = pd.to_numeric(df_user_behavior['评分'], errors='coerce')
# 确保价格是数值型
df_user_behavior['商品价格'] = pd.to_numeric(df_user_behavior['商品价格'], errors='coerce')
df_product_info['价格'] = pd.to_numeric(df_product_info['价格'], errors='coerce')

# 处理可能的缺失评分（用平均分填充，或删除）
# 这里我们选择删除评分为空的行
df_user_behavior.dropna(subset=['评分'], inplace=True)

print("--- 数据清洗完成 ---")

# 3. 特征工程
# 构建用户-商品评分矩阵 (用于协同过滤)
user_item_matrix = df_user_behavior.pivot_table(index='用户ID', columns='商品ID', values='评分', fill_value=0)
print("--- 用户-商品评分矩阵构建完成 ---")

# 4. 简单分析
# a. 最受欢迎的商品 (按平均评分和销量)
top_rated_products = df_product_info.sort_values(by='平均评分', ascending=False)[['商品ID', '商品名称', '平均评分']].head(10)
top_sold_products = df_product_info.sort_values(by='销量', ascending=False)[['商品ID', '商品名称', '销量']].head(10)

# b. 用户行为分析
user_rating_distribution = df_user_behavior['评分'].value_counts().sort_index()
user_activity = df_user_behavior.groupby('用户ID')['行为类型'].count().sort_values(ascending=False)
item_popularity = df_user_behavior['商品ID'].value_counts().sort_values(ascending=False)

print("--- 简单分析完成 ---")

# 5. 准备协同过滤模型 (基于用户的协同过滤)
# 选择评分矩阵中的前100个用户进行演示
sample_users = user_item_matrix.index[:100]
user_item_matrix_sample = user_item_matrix.loc[sample_users]

# 标准化评分矩阵
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix_sample.T).T # 转置后标准化用户向量
user_item_matrix_scaled = pd.DataFrame(user_item_matrix_scaled, index=user_item_matrix_sample.index, columns=user_item_matrix_sample.columns)

# 计算用户相似度 (使用标准化后的数据)
user_similarity_matrix = cosine_similarity(user_item_matrix_scaled)
user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_item_matrix_sample.index, columns=user_item_matrix_sample.index)

print("--- 用户相似度矩阵计算完成 ---")

# 6. 简单的基于用户协同过滤推荐示例
def recommend_items_for_user(user_id, user_item_matrix, user_similarity_df, n_recommendations=5):
    """
    为指定用户推荐商品
    :param user_id: 目标用户ID
    :param user_item_matrix: 用户-商品评分矩阵
    :param user_similarity_df: 用户相似度矩阵
    :param n_recommendations: 推荐商品数量
    :return: 推荐商品DataFrame
    """
    if user_id not in user_item_matrix.index:
        print(f"用户 {user_id} 不在数据集中。")
        return pd.DataFrame(columns=['商品ID', '商品名称'])

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
    recommended_item_names = df_product_info[df_product_info['商品ID'].isin(recommended_item_ids)][['商品ID', '商品名称']].copy()
    # 添加推荐得分
    score_dict = dict(recommended_items)
    recommended_item_names['推荐得分'] = recommended_item_names['商品ID'].map(score_dict)
    
    return recommended_item_names

# 为用户 U001 推荐商品
target_user = 'U001'
recommended_items_df = recommend_items_for_user(target_user, user_item_matrix_sample, user_similarity_df, n_recommendations=5)
print(f"--- 为用户 {target_user} 推荐的商品 ---")
print(recommended_items_df)

# 7. 保存结果到文件
# 保存热门商品分析结果
top_rated_products.to_csv('E:/datalogic-main/08 推荐系统/分析结果_高评分商品.csv', index=False, encoding='utf-8-sig')
top_sold_products.to_csv('E:/datalogic-main/08 推荐系统/分析结果_高销量商品.csv', index=False, encoding='utf-8-sig')

# 保存用户行为分析结果
user_rating_distribution.to_csv('E:/datalogic-main/08 推荐系统/分析结果_用户评分分布.csv', header=['次数'], encoding='utf-8-sig')
user_activity.to_csv('E:/datalogic-main/08 推荐系统/分析结果_用户活跃度.csv', header=['行为次数'], encoding='utf-8-sig')
item_popularity.to_csv('E:/datalogic-main/08 推荐系统/分析结果_商品热度.csv', header=['被交互次数'], encoding='utf-8-sig')

# 保存为用户 U001 的推荐结果
recommended_items_df.to_csv('E:/datalogic-main/08 推荐系统/推荐结果_用户U001.csv', index=False, encoding='utf-8-sig')

print("--- 分析结果已保存至CSV文件 ---")