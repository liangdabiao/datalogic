import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与预处理
user_behavior_path = 'E:/datalogic-main/08 推荐系统/用户行为数据.csv'
product_info_path = 'E:/datalogic-main/08 推荐系统/商品信息数据.csv'

df_user_behavior = pd.read_csv(user_behavior_path, encoding='utf-8')
df_product_info = pd.read_csv(product_info_path, encoding='utf-8')

df_user_behavior['评分'] = pd.to_numeric(df_user_behavior['评分'], errors='coerce')
df_user_behavior.dropna(subset=['评分'], inplace=True)

user_item_matrix = df_user_behavior.pivot_table(index='用户ID', columns='商品ID', values='评分', fill_value=0)

# 2. 模型评估 (使用留一法 LOO)
def evaluate_user_based_cf(user_item_matrix, top_k=5, num_users_to_eval=50, num_users_for_similarity=100):
    """
    使用留一法评估基于用户的协同过滤模型
    :param user_item_matrix: 用户-商品评分矩阵 (完整)
    :param top_k: 推荐列表长度
    :param num_users_to_eval: 要评估的用户数量
    :param num_users_for_similarity: 用于计算相似度的用户样本数量
    :return: 平均精确率@K, 平均召回率@K
    """
    precisions = []
    recalls = []
    
    # 选择要评估的用户
    users_to_evaluate = user_item_matrix.index[:num_users_to_eval] 
    
    # 选择用于计算相似度的用户样本
    sample_users = user_item_matrix.index[:num_users_for_similarity]
    user_item_matrix_sample = user_item_matrix.loc[sample_users]
    
    # 对用户-商品矩阵进行标准化 (只对样本用户)
    scaler = StandardScaler()
    # 注意：需要处理fit和transform的列对齐问题
    user_item_matrix_scaled_np = scaler.fit_transform(user_item_matrix_sample)
    user_item_matrix_scaled = pd.DataFrame(user_item_matrix_scaled_np, index=user_item_matrix_sample.index, columns=user_item_matrix_sample.columns)

    print(f"开始评估 {len(users_to_evaluate)} 个用户...")

    for i, user_id in enumerate(users_to_evaluate):
        if user_id not in user_item_matrix.index:
            continue
            
        user_ratings = user_item_matrix.loc[user_id]
        # 找到该用户所有有评分的商品
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) < 2:
            # 如果用户评分的商品少于2个，无法进行留一法评估
            continue
            
        # 选择一个商品作为测试集 (这里选择评分最高的商品之一，模拟“重要”商品)
        # 更严格的做法是按时间选择最后一个
        test_item = rated_items.idxmax() 
        # test_item = np.random.choice(rated_items.index) # 或者随机选择
        train_items = rated_items.drop(test_item).index.tolist()
        
        # 构造该用户的训练集评分向量 (测试商品评分为0)
        user_train_ratings = user_ratings.copy()
        user_train_ratings[test_item] = 0 
        
        # ------------------- 修正部分 -------------------
        # 为了计算相似度，我们需要将 user_train_ratings 对齐到标准化的特征空间
        # 1. 确保 user_train_ratings 的列与 scaler 训练时的列一致
        aligned_user_vector = user_train_ratings.reindex(user_item_matrix_scaled.columns, fill_value=0).values.reshape(1, -1)
        
        # 2. 使用之前 fit 好的 scaler 进行 transform
        # 注意：这里直接用 numpy 数组，因为 reindex 已经保证了列对齐
        user_vector_scaled = scaler.transform(aligned_user_vector)
        # ------------------------------------------------
        
        # 计算当前用户与样本中其他用户的相似度
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
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == len(users_to_evaluate):
            print(f"已评估 {i + 1}/{len(users_to_evaluate)} 个用户...")

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    return avg_precision, avg_recall, len(precisions)

print("--- 开始评估模型... ---")
# 执行评估
avg_prec, avg_rec, num_evaluated = evaluate_user_based_cf(user_item_matrix, top_k=5, num_users_to_eval=30, num_users_for_similarity=100)
print(f"--- 模型评估完成 (评估了 {num_evaluated} 个用户)---")
print(f"平均精确率@5 (P@5): {avg_prec:.4f}")
print(f"平均召回率@5 (R@5): {avg_rec:.4f}")

# 3. 保存评估结果
evaluation_result = pd.DataFrame({
    'Metric': ['Precision@5', 'Recall@5', 'Users_Evaluated'],
    'Value': [avg_prec, avg_rec, num_evaluated]
})
evaluation_result.to_csv('E:/datalogic-main/08 推荐系统/模型评估结果_基于用户协同过滤.csv', index=False, encoding='utf-8-sig')
print("--- 模型评估结果已保存至 CSV 文件 ---")