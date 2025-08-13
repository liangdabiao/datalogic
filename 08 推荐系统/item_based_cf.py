import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与预处理 (同上)
user_behavior_path = 'E:/datalogic-main/08 推荐系统/用户行为数据.csv'
product_info_path = 'E:/datalogic-main/08 推荐系统/商品信息数据.csv'

df_user_behavior = pd.read_csv(user_behavior_path, encoding='utf-8')
df_product_info = pd.read_csv(product_info_path, encoding='utf-8')

df_user_behavior['评分'] = pd.to_numeric(df_user_behavior['评分'], errors='coerce')
df_user_behavior.dropna(subset=['评分'], inplace=True)

user_item_matrix = df_user_behavior.pivot_table(index='用户ID', columns='商品ID', values='评分', fill_value=0)

# 转置得到 商品-用户 矩阵，用于基于物品的协同过滤
item_user_matrix = user_item_matrix.T

# 2. 基于物品的协同过滤推荐函数
def recommend_items_item_based(user_id, user_item_matrix, item_user_matrix, item_similarity_df, top_k=5, n_recommendations=5):
    """
    基于物品的协同过滤推荐
    :param user_id: 目标用户ID
    :param user_item_matrix: 用户-商品评分矩阵
    :param item_user_matrix: 商品-用户评分矩阵
    :param item_similarity_df: 商品相似度矩阵
    :param top_k: 考虑用户评分过的商品中，最相似的前K个
    :param n_recommendations: 推荐商品数量
    :return: 推荐商品DataFrame
    """
    if user_id not in user_item_matrix.index:
        print(f"用户 {user_id} 不在数据集中。")
        return pd.DataFrame(columns=['商品ID', '商品名称'])

    # 获取用户评分过的商品
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]

    if rated_items.empty:
        print(f"用户 {user_id} 没有评分记录。")
        return pd.DataFrame(columns=['商品ID', '商品名称'])

    # 存储所有可能的推荐商品及其预测评分
    recommendations = {}

    # 遍历用户评分过的每个商品
    for item_id, rating in rated_items.items():
        if item_id not in item_similarity_df.index:
            continue
            
        # 找到与当前商品最相似的其他商品
        similar_items = item_similarity_df[item_id].drop(item_id, errors='ignore').sort_values(ascending=False).head(top_k)
        
        # 基于相似度和用户评分，计算对相似商品的预测评分
        for similar_item_id, similarity in similar_items.items():
            if similar_item_id not in recommendations:
                # 累计相似度 * 评分 作为预测得分
                recommendations[similar_item_id] = similarity * rating

    # 移除用户已经评分过的商品
    for item_id in rated_items.index:
        recommendations.pop(item_id, None) # 使用pop避免KeyError

    # 按预测得分排序并返回Top N
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # 获取商品名称
    recommended_item_ids = [item[0] for item in sorted_recommendations]
    recommended_item_names = df_product_info[df_product_info['商品ID'].isin(recommended_item_ids)][['商品ID', '商品名称']].copy()
    # 添加预测得分
    score_dict = dict(sorted_recommendations)
    recommended_item_names['预测得分'] = recommended_item_names['商品ID'].map(score_dict)
    
    return recommended_item_names


# 3. 模型评估 (使用留一法 LOO for Item-Based CF)
def evaluate_item_based_cf(user_item_matrix, item_user_matrix, top_k=5, n_recommendations=5, num_users_to_eval=50):
    """
    使用留一法评估基于物品的协同过滤模型
    :param user_item_matrix: 用户-商品评分矩阵
    :param item_user_matrix: 商品-用户评分矩阵
    :param top_k: 计算相似商品时考虑的邻居数
    :param n_recommendations: 推荐列表长度
    :param num_users_to_eval: 要评估的用户数量
    :return: 平均精确率@K, 平均召回率@K
    """
    precisions = []
    recalls = []
    
     # 选择要评估的用户
    users_to_evaluate = user_item_matrix.index[:num_users_to_eval] 
    
    print(f"开始评估 {len(users_to_evaluate)} 个用户 (基于物品的协同过滤)...")

    # 预先计算所有商品的相似度矩阵 (这是一个耗时操作，只做一次)
    print("正在计算商品相似度矩阵...")
    # 为了效率，也只使用部分商品进行计算
    sample_items = item_user_matrix.index[:150] # 限制商品数量
    item_user_matrix_sample = item_user_matrix.loc[sample_items]
    
    scaler = StandardScaler()
    item_user_matrix_scaled_np = scaler.fit_transform(item_user_matrix_sample)
    item_user_matrix_scaled = pd.DataFrame(item_user_matrix_scaled_np, index=item_user_matrix_sample.index, columns=item_user_matrix_sample.columns)
    
    item_similarity_matrix = cosine_similarity(item_user_matrix_scaled)
    item_similarity_df = pd.DataFrame(item_similarity_matrix, index=item_user_matrix_sample.index, columns=item_user_matrix_sample.index)
    print("商品相似度矩阵计算完成。")
    
    for i, user_id in enumerate(users_to_evaluate):
        if user_id not in user_item_matrix.index:
            continue
            
        user_ratings = user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) < 2:
            continue
            
        # 选择一个商品作为测试集 (这里选择评分最高的)
        test_item = rated_items.idxmax() 
        train_items = rated_items.drop(test_item).index.tolist()
        
        # 构造该用户的训练集评分向量 (测试商品评分为0)
        user_train_ratings = user_ratings.copy()
        user_train_ratings[test_item] = 0 
        
        # 创建一个临时的、用于推荐的用户-商品矩阵 (只修改当前用户那一行)
        temp_user_item_matrix = user_item_matrix.copy()
        temp_user_item_matrix.loc[user_id] = user_train_ratings
        
        # 使用基于物品的协同过滤进行推荐
        # 注意：这里传入的是原始的、未修改的矩阵来获取用户历史评分
        recommended_df = recommend_items_item_based(user_id, user_item_matrix, item_user_matrix, item_similarity_df, top_k, n_recommendations)
        recommended_item_ids = set(recommended_df['商品ID'].tolist())
        
        # 计算Precision和Recall
        relevant_items = {test_item}
        hits = recommended_item_ids.intersection(relevant_items)
        num_hits = len(hits)
        
        precision = num_hits / n_recommendations if n_recommendations > 0 else 0.0
        recall = num_hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == len(users_to_evaluate):
            print(f"已评估 {i + 1}/{len(users_to_evaluate)} 个用户...")

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    num_evaluated = len(precisions)
    
    return avg_precision, avg_recall, num_evaluated


# --- 主程序 ---
print("--- 开始评估基于物品的协同过滤模型... ---")
# 执行评估
avg_prec_item, avg_rec_item, num_evaluated_item = evaluate_item_based_cf(
    user_item_matrix, item_user_matrix, 
    top_k=10, n_recommendations=5, num_users_to_eval=30
)
print(f"--- 基于物品的协同过滤模型评估完成 (评估了 {num_evaluated_item} 个用户)---")
print(f"平均精确率@5 (P@5): {avg_prec_item:.4f}")
print(f"平均召回率@5 (R@5): {avg_rec_item:.4f}")

# 4. 为用户 U001 生成一个基于物品的推荐示例
# 预先计算商品相似度矩阵 (用于示例)
print("正在为示例计算商品相似度矩阵...")
sample_items_full = item_user_matrix.index[:100] # 为示例用更小的集
item_user_matrix_sample_full = item_user_matrix.loc[sample_items_full]
scaler_full = StandardScaler()
item_user_matrix_scaled_np_full = scaler_full.fit_transform(item_user_matrix_sample_full)
item_user_matrix_scaled_full = pd.DataFrame(item_user_matrix_scaled_np_full, index=item_user_matrix_sample_full.index, columns=item_user_matrix_sample_full.columns)
item_similarity_matrix_full = cosine_similarity(item_user_matrix_scaled_full)
item_similarity_df_full = pd.DataFrame(item_similarity_matrix_full, index=item_user_matrix_sample_full.index, columns=item_user_matrix_sample_full.index)
print("示例商品相似度矩阵计算完成。")

target_user = 'U001'
recommended_items_item_based_df = recommend_items_item_based(target_user, user_item_matrix, item_user_matrix, item_similarity_df_full, top_k=5, n_recommendations=5)
print(f"\n--- 基于物品的协同过滤: 为用户 {target_user} 推荐的商品 ---")
print(recommended_items_item_based_df)

# 5. 保存评估结果和示例推荐结果
evaluation_result_item = pd.DataFrame({
    'Metric': ['Precision@5', 'Recall@5', 'Users_Evaluated'],
    'Value': [avg_prec_item, avg_rec_item, num_evaluated_item],
    'Model': ['Item-Based CF'] * 3
})

# 读取之前用户协同过滤的评估结果进行比较
try:
    eval_user_cf = pd.read_csv('E:/datalogic-main/08 推荐系统/模型评估结果_基于用户协同过滤.csv')
    eval_user_cf['Model'] = 'User-Based CF'
    # 合并评估结果
    combined_eval = pd.concat([eval_user_cf, evaluation_result_item], ignore_index=True)
    combined_eval.to_csv('E:/datalogic-main/08 推荐系统/模型评估结果_综合比较.csv', index=False, encoding='utf-8-sig')
    print("--- 综合模型评估结果已保存至 CSV 文件 ---")
except FileNotFoundError:
    print("未找到基于用户的协同过滤评估结果文件，仅保存当前结果。")
    evaluation_result_item.to_csv('E:/datalogic-main/08 推荐系统/模型评估结果_基于物品协同过滤.csv', index=False, encoding='utf-8-sig')

# 保存示例推荐结果
recommended_items_item_based_df.to_csv('E:/datalogic-main/08 推荐系统/推荐结果_用户U001_基于物品协同过滤.csv', index=False, encoding='utf-8-sig')
print("--- 用户U001的基于物品协同过滤推荐结果已保存至 CSV 文件 ---")