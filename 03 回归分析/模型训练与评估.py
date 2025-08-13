import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    """准备建模数据"""
    print("=== 数据准备与特征工程 ===")
    
    # 创建特征
    df['房龄'] = 2025 - df['建造年份']
    df['楼层比例'] = df['楼层'] / df['总楼层']
    df['房间密度'] = df['面积'] / df['房间数'].replace(0, 1)
    df['距离总分'] = df['地铁距离'] + df['学校距离'] + df['商场距离']
    
    # 处理分类变量
    decoration_map = {'毛坯': 1, '简装修': 2, '精装修': 3, '豪华装修': 4}
    estate_map = {'普通小区': 1, '高档小区': 2, '豪华小区': 3}
    direction_map = {'南': 4, '东': 3, '西': 2, '北': 1}
    
    df['装修等级_num'] = df['装修等级'].map(decoration_map)
    df['小区类型_num'] = df['小区类型'].map(estate_map)
    df['朝向_num'] = df['朝向'].map(direction_map)
    
    # 特征列表
    feature_cols = [
        '面积', '房间数', '卫生间数', '房龄', '楼层比例', 
        '房间密度', '距离总分', '装修等级_num', '小区类型_num', '朝向_num'
    ]
    
    X = df[feature_cols]
    y = df['房价']
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    
    return X, y, feature_cols

def train_and_evaluate_models(X, y):
    """训练并评估多种模型"""
    print("\n=== 模型训练与评估 ===")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # 训练模型
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'model': model
        }
        
        print(f"R² Score (Train/Test): {train_r2:.3f} / {test_r2:.3f}")
        print(f"MAE (Train/Test): {train_mae:.0f} / {test_mae:.0f}")
        print(f"RMSE (Train/Test): {train_rmse:.0f} / {test_rmse:.0f}")
    
    return results, scaler

def feature_importance_analysis(X, y, feature_cols):
    """特征重要性分析"""
    print("\n=== 特征重要性分析 ===")
    
    # 使用随机森林获取特征重要性
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("特征重要性排序:")
    for i, row in feature_importance.iterrows():
        ranking = feature_importance.index.get_loc(i) + 1
        print(f"{ranking}. {row['feature']}: {row['importance']:.3f}")
    
    return feature_importance

def generate_predictions_example(X, y, model, scaler):
    """生成预测示例"""
    print("\n=== 预测示例 ===")
    
    # 使用线性回归模型
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # 创建一些示例预测
    sample_data = [
        # [面积, 房间数, 卫生间数, 房龄, 楼层比例, 房间密度, 距离总分, 装修等级_num, 小区类型_num, 朝向_num]
        [80, 2, 1, 10, 0.3, 40, 3.0, 3, 2, 4],  # 普通精装修住宅
        [150, 4, 2, 5, 0.2, 37.5, 1.0, 4, 3, 4],  # 豪华精装修住宅
        [70, 2, 1, 20, 0.5, 35, 5.0, 2, 1, 2],  # 老旧简装修住宅
    ]
    
    sample_names = ['中档住宅', '豪华住宅', '经济住宅']
    
    for name, data in zip(sample_names, sample_data):
        data_scaled = scaler.transform([data])
        predicted_price = lr_model.predict(data_scaled)[0]
        print(f"{name}: 预测价格 {predicted_price:,.0f} 元")
    
    return lr_model

# 主函数执行
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv('房价数据_分析版.csv', encoding='utf-8-sig')
    
    # 数据准备
    X, y, feature_cols = prepare_data(df)
    
    # 模型训练与评估
    results, scaler = train_and_evaluate_models(X, y)
    
    # 特征重要性分析
    feature_importance = feature_importance_analysis(X, y, feature_cols)
    
    # 生成预测示例
    generate_predictions_example(X, y, results['Linear Regression']['model'], scaler)
    
    # 总结报告
    print("\n=== 总结报告 ===")
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"最佳模型: {best_model[0]} (R² = {best_model[1]['test_r2']:.3f})")
    
    # 保存模型评估结果
    results_df = pd.DataFrame(results).T
    results_df = results_df.drop('model', axis=1)

    # 打印模型评估结果
    print(f"\n打印模型评估结果:")
    print(results_df)
    
    print(f"\n各模型性能总结:")
    for model_name, metrics in results.items():
        print(f"{model_name}: R²={metrics['test_r2']:.3f}, MAE={metrics['test_mae']:.0f}, RMSE={metrics['test_rmse']:.0f}")
    
    print("\n分析完成！房价预测模型已建立成功。")







#     === 数据准备与特征工程 ===
# 特征数量: 10
# 特征列表: ['面积', '房间数', '卫生间数', '房龄', '楼层比例', '房间密度', '距离总分', '装修等级_num', '小区类型_num', '朝向_num']

# === 模型训练与评估 ===

# --- Linear Regression ---
# R² Score (Train/Test): 0.981 / 0.957
# MAE (Train/Test): 50947 / 66139
# RMSE (Train/Test): 66105 / 80367

# --- Random Forest ---
# R² Score (Train/Test): 0.999 / 0.990
# MAE (Train/Test): 10074 / 23415
# RMSE (Train/Test): 17526 / 39281

# --- Decision Tree ---
# R² Score (Train/Test): 1.000 / 0.949
# MAE (Train/Test): 0 / 51000
# RMSE (Train/Test): 0 / 87807

# === 特征重要性分析 ===
# 特征重要性排序:
# 1. 面积: 0.452
# 2. 房间数: 0.267
# 3. 距离总分: 0.125
# 4. 房龄: 0.075
# 5. 装修等级_num: 0.040
# 6. 小区类型_num: 0.028
# 7. 房间密度: 0.008
# 8. 卫生间数: 0.004
# 9. 楼层比例: 0.002
# 10. 朝向_num: 0.000

# === 预测示例 ===
# 中档住宅: 预测价格 319,560 元
# 豪华住宅: 预测价格 1,215,742 元
# 经济住宅: 预测价格 107,691 元

# === 总结报告 ===
# 最佳模型: Random Forest (R² = 0.990)

# 打印模型评估结果:
#                    train_r2   test_r2     train_mae      test_mae    train_rmse     test_rmse
# Linear Regression  0.980844    0.9575  50947.168036  66139.267765  66104.759114  80366.625321
# Random Forest      0.998653  0.989847      10073.75       23415.0  17525.927222  39281.274165
# Decision Tree           1.0  0.949266           0.0       51000.0           0.0  87806.605674

# 各模型性能总结:
# Linear Regression: R²=0.957, MAE=66139, RMSE=80367
# Random Forest: R²=0.990, MAE=23415, RMSE=39281
# Decision Tree: R²=0.949, MAE=51000, RMSE=87807

# 分析完成！房价预测模型已建立成功。
