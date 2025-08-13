import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 数据加载与预处理 (同上一个脚本)
user_behavior_path = 'E:/datalogic-main/08 推荐系统/用户行为数据.csv'
product_info_path = 'E:/datalogic-main/08 推荐系统/商品信息数据.csv'

df_user_behavior = pd.read_csv(user_behavior_path, encoding='utf-8')
df_product_info = pd.read_csv(product_info_path, encoding='utf-8')

df_user_behavior['评分'] = pd.to_numeric(df_user_behavior['评分'], errors='coerce')
df_user_behavior.dropna(subset=['评分'], inplace=True)

user_item_matrix = df_user_behavior.pivot_table(index='用户ID', columns='商品ID', values='评分', fill_value=0)

# 2. 模型评估 (使用留一法 LOO)
def evaluate_user_based_cf(user_item_matrix, top_k=5):
    """
    使用留一法评估基于用户的协同过滤模型
    :param user_item_matrix: 用户-商品评分矩阵
    :param top_k: 推荐列表长度
    :return: 平均精确率@K, 平均召回率@K
    """
    precisions = []
    recalls = []
    
    # 为了简化计算，我们只评估一部分用户 (例如前50个)
    # 全量评估会非常耗时
    users_to_evaluate = user_item_matrix.index[:50] 
    
    for user_id in users_to_evaluate:
        user_ratings = user_item_matrix.loc[user_id]
        # 找到该用户所有有评分的商品
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) < 2:
            # 如果用户评分的商品少于2个，无法进行留一法评估
            continue
            
        # 随机选择一个商品作为测试集
        # 在实际应用中，通常选择时间上最新的一个
        test_item = np.random.choice(rated_items.index)
        train_items = rated_items.drop(test_item).index.tolist()
        
        # 构造该用户的训练集评分向量
        user_train_ratings = user_ratings.copy()
        user_train_ratings[test_item] = 0 # 将测试商品的评分设为0
        
        # 基于训练集评分向量，寻找相似用户并进行推荐
        # 这里简化处理，直接用整个训练矩阵计算相似度
        # 更高效的方法是只计算与当前用户相关的部分
        
        # 计算用户相似度 (为了效率，这里也只用部分用户)
        sample_users = user_item_matrix.index[:100] # 限制用户数量以提高速度
        user_item_matrix_sample = user_item_matrix.loc[sample_users]
        
        scaler = StandardScaler()
        user_item_matrix_scaled = scaler.fit_transform(user_item_matrix_sample.T).T
        user_item_matrix_scaled = pd.DataFrame(user_item_matrix_scaled, index=user_item_matrix_sample.index, columns=user_item_matrix_sample.columns)
        
        # 计算当前用户与样本中其他用户的相似度
        user_vector = user_train_ratings.loc[user_item_matrix_sample.columns].values.reshape(1, -1)
        user_vector_scaled = scaler.transform(user_vector)
        user_similarities = cosine_similarity(user_vector_scaled, user_item_matrix_scaled)[0]
        user_similarity_series = pd.Series(user_similarities, index=user_item_matrix_sample.index)
        
        # 获取相似用户（排除自己）
        similar_users = user_similarity_series.drop(user_id, errors='ignore').sort_values(ascending=False)
        
        # 收集相似用户喜欢的商品，并进行加权求和
        recommendations = {}
        user_rated_item_ids = set(train_items) # 训练集中已评分的商品
        for similar_user_id, similarity_score in similar_users.items():
            if similarity_score <= 0: continue # 只考虑正相关的用户
            similar_user_ratings = user_item_matrix_sample.loc[similar_user_id]
            for item_id, rating in similar_user_ratings.items():
                if item_id not in user_rated_item_ids and rating > 0:
                    if item_id in recommendations:
                        recommendations[item_id] += similarity_score * rating
                    else:
                        recommendations[item_id] = similarity_score * rating
        
        # 获取推荐列表
        recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k]
        recommended_item_ids = set([item[0] for item in recommended_items])
        
        # 计算Precision和Recall
        # Relevant items (在测试集中): {test_item}
        # Recommended items: recommended_item_ids
        relevant_items = {test_item}
        
        hits = recommended_item_ids.intersection(relevant_items)
        num_hits = len(hits)
        
        precision = num_hits / top_k if top_k > 0 else 0.0
        recall = num_hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        
        # 为了控制台能看到进度
        if len(precisions) % 10 == 0:
            print(f"已评估 {len(precisions)} 个用户...")
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    return avg_precision, avg_recall

print("--- 开始评估模型... ---")
# 执行评估 (这可能需要一些时间)
avg_prec, avg_rec = evaluate_user_based_cf(user_item_matrix, top_k=5)
print(f"--- 模型评估完成 ---")
print(f"平均精确率@5 (P@5): {avg_prec:.4f}")
print(f"平均召回率@5 (R@5): {avg_rec:.4f}")

# 3. 保存评估结果
evaluation_result = pd.DataFrame({
    'Metric': ['Precision@5', 'Recall@5'],
    'Value': [avg_prec, avg_rec]
})
evaluation_result.to_csv('E:/datalogic-main/08 推荐系统/模型评估结果_基于用户协同过滤.csv', index=False, encoding='utf-8-sig')
print("--- 模型评估结果已保存至 CSV 文件 ---")