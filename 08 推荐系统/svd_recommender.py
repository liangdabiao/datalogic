import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 1. 数据加载与预处理 (同上)
user_behavior_path = 'E:/datalogic-main/08 推荐系统/用户行为数据.csv'
product_info_path = 'E:/datalogic-main/08 推荐系统/商品信息数据.csv'

df_user_behavior = pd.read_csv(user_behavior_path, encoding='utf-8')
df_product_info = pd.read_csv(product_info_path, encoding='utf-8')

df_user_behavior['评分'] = pd.to_numeric(df_user_behavior['评分'], errors='coerce')
df_user_behavior.dropna(subset=['评分'], inplace=True)

user_item_matrix = df_user_behavior.pivot_table(index='用户ID', columns='商品ID', values='评分', fill_value=0)

# 2. SVD 矩阵分解推荐函数
class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.is_fitted = False

    def fit(self, user_item_matrix):
        """
        训练 SVD 模型
        :param user_item_matrix: 用户-商品评分矩阵
        """
        self.user_item_matrix = user_item_matrix
        
        # 创建索引映射
        self.user_to_idx = {user: i for i, user in enumerate(user_item_matrix.index)}
        self.idx_to_user = {i: user for user, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(user_item_matrix.columns)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}
        
        # 执行 SVD 分解
        self.user_factors = self.svd.fit_transform(user_item_matrix)
        self.item_factors = self.svd.components_.T # 转置以对齐维度
        
        self.is_fitted = True
        print(f"SVD 模型训练完成。用户因子维度: {self.user_factors.shape}, 商品因子维度: {self.item_factors.shape}")

    def predict(self, user_id, item_id):
        """
        预测用户对商品的评分
        :param user_id: 用户ID
        :param item_id: 商品ID
        :return: 预测评分
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法。")
        
        try:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            # 计算点积得到预测评分
            predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            return predicted_rating
        except KeyError:
            # 如果用户或商品不在训练集中，则返回平均评分或0
            return self.user_item_matrix.mean().mean() 

    def recommend(self, user_id, n_recommendations=5):
        """
        为用户推荐商品
        :param user_id: 用户ID
        :param n_recommendations: 推荐商品数量
        :return: 推荐商品DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法。")
            
        if user_id not in self.user_to_idx:
             print(f"警告: 用户 {user_id} 不在训练数据中。")
             # 可以返回热门商品或空列表
             return pd.DataFrame(columns=['商品ID', '商品名称', '预测得分'])

        user_idx = self.user_to_idx[user_id]
        # 获取用户对所有商品的预测评分
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        # 计算该用户对所有商品的预测评分
        # user_factors[user_idx] shape: (1, n_components)
        # item_factors shape: (n_items, n_components)
        # predicted_ratings shape: (1, n_items) -> (n_items,)
        predicted_ratings = np.dot(self.user_factors[user_idx], self.item_factors.T)
        
        # 创建一个Series来存储预测评分，并对齐商品ID
        predictions = pd.Series(predicted_ratings, index=self.user_item_matrix.columns)
        
        # 过滤掉用户已经评分过的商品
        recommendations = predictions.drop(rated_items, errors='ignore').sort_values(ascending=False).head(n_recommendations)
        
        # 获取商品名称和预测得分
        recommended_items_df = df_product_info[df_product_info['商品ID'].isin(recommendations.index)][['商品ID', '商品名称']].copy()
        recommended_items_df['预测得分'] = recommended_items_df['商品ID'].map(recommendations.to_dict())
        
        return recommended_items_df


# 3. 使用 SVD 模型
print("--- 开始训练 SVD 矩阵分解模型 ---")
# 初始化并训练模型
svd_recommender = SVDRecommender(n_components=20) # 组件数不宜过高，避免过拟合
svd_recommender.fit(user_item_matrix)

# 为用户 U001 生成推荐
target_user = 'U001'
recommended_items_svd_df = svd_recommender.recommend(target_user, n_recommendations=5)
print(f"\n--- SVD 矩阵分解: 为用户 {target_user} 推荐的商品 ---")
print(recommended_items_svd_df)

# 4. 保存 SVD 推荐结果
recommended_items_svd_df.to_csv('E:/datalogic-main/08 推荐系统/推荐结果_用户U001_SVD矩阵分解.csv', index=False, encoding='utf-8-sig')
print("--- 用户U001的SVD矩阵分解推荐结果已保存至 CSV 文件 ---")

# 5. (可选) 简单评估 SVD 模型
# 由于SVD是降维重构，通常不直接用于留一法评估预测准确性，
# 但可以计算重构误差 (MSE) 作为模型拟合度的一个指标
def calculate_mse(original_matrix, user_factors, item_factors):
    """计算原始矩阵与重构矩阵之间的均方误差 (MSE)"""
    reconstructed = np.dot(user_factors, item_factors.T)
    mse = np.mean((original_matrix - reconstructed) ** 2)
    return mse

# 将 pandas DataFrame 转换为 numpy array 用于计算
original_matrix_np = user_item_matrix.values
mse = calculate_mse(original_matrix_np, svd_recommender.user_factors, svd_recommender.item_factors)
print(f"\n--- SVD 模型重构误差 (MSE): {mse:.4f} ---")

# 保存 MSE 结果
mse_result = pd.DataFrame({'Metric': ['MSE'], 'Value': [mse]})
mse_result.to_csv('E:/datalogic-main/08 推荐系统/模型评估结果_SVD_MSE.csv', index=False, encoding='utf-8-sig')
print("--- SVD 模型 MSE 评估结果已保存至 CSV 文件 ---")