import pandas as pd
import numpy as np

# 数据加载与清洗
def load_and_clean_data():
    print("=== 1. 数据加载与清洗 ===")
    
    # 读取数据
    df = pd.read_csv('房价预测数据.csv', skiprows=1, header=None, encoding='utf-8')
    
    # 手动设置正确的列名
    columns = ['房屋ID', '面积', '房间数', '卫生间数', '楼层', '总楼层', 
               '建造年份', '地铁距离', '学校距离', '商场距离', 
               '装修等级', '朝向', '小区类型', '房价']
    df.columns = columns
    
    # 数据类型转换
    numeric_cols = ['面积', '房间数', '卫生间数', '楼层', '总楼层', 
                   '建造年份', '地铁距离', '学校距离', '商场距离', '房价']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"数据集形状: {df.shape}")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print(f"\n缺失值统计:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "无缺失值")
    
    # 检测异常值（使用IQR方法）
    def detect_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    outliers = []
    for col in numeric_cols:
        outlier_mask = detect_outliers(df[col])
        if outlier_mask.any():
            outliers.append(f"{col}: {outlier_mask.sum()}个异常值")
    
    print(f"\n异常值检测:")
    print(outliers if outliers else "无异常值")
    
    # 显示数据摘要
    print(f"\n数值统计摘要:")
    print(df[numeric_cols].describe())
    
    # 显示分类特征分布
    print(f"\n装修等级分布:")
    print(df['装修等级'].value_counts())
    
    print(f"\n小区类型分布:")
    print(df['小区类型'].value_counts())
    
    print(f"\n朝向分布:")
    print(df['朝向'].value_counts())
    
    return df

def perform_correlation_analysis(df):
    """相关性分析"""
    print("\n=== 2. 相关性分析 ===")
    
    # 计算房龄
    df['房龄'] = 2025 - df['建造年份']
    
    # 创建特征工程变量
    df['楼层比例'] = df['楼层'] / df['总楼层']
    df['房间密度'] = df['面积'] / df['房间数'].replace(0, 1)  # 避免除以零
    df['距离总分'] = df['地铁距离'] + df['学校距离'] + df['商场距离']
    
    # 数值化分类变量
    decoration_map = {'毛坯': 1, '简装修': 2, '精装修': 3, '豪华装修': 4}
    estate_map = {'普通小区': 1, '高档小区': 2, '豪华小区': 3}
    
    df['装修等级_num'] = df['装修等级'].map(decoration_map)
    df['小区类型_num'] = df['小区类型'].map(estate_map)
    
    # 计算相关性
    analysis_cols = ['面积', '房间数', '卫生间数', '房龄', '楼层比例', 
                    '房间密度', '距离总分', '装修等级_num', '小区类型_num', '房价']
    
    correlation_matrix = df[analysis_cols].corr()
    
    print("房价与各个特征的相关性（前5个）:")
    price_corr = correlation_matrix['房价'].sort_values(ascending=False)
    print(price_corr.head())
    
    # 找出正相关和负相关的特征
    positive_corr = price_corr[price_corr > 0.6]
    negative_corr = price_corr[price_corr < -0.3]
    
    print(f"\n强正相关特征 (>0.6):")
    print(positive_corr.drop('房价')) if len(positive_corr) > 1 else print("无")
    
    print(f"\n负相关特征 (<-0.3):")
    print(negative_corr) if len(negative_corr) > 0 else print("无")
    
    return df

if __name__ == "__main__":
    # 数据加载与清洗
    df = load_and_clean_data()
    
    # 相关性分析
    df_analysis = perform_correlation_analysis(df)
    
    # 保存分析后的数据
    df_analysis.to_csv('房价数据_分析版.csv', index=False, encoding='utf-8-sig')
    print(f"\n分析完成！数据已保存至: 房价数据_分析版.csv")
    
    print(f"\n关键发现:")
    print(f"- 样本量: {len(df)}条记录")
    print(f"- 价格范围: {df['房价'].min():.0f}万 - {df['房价'].max():.0f}万")
    print(f"- 平均价格: {df['房价'].mean():.0f}万")
    print(f"- 数据准备就绪，可进入下一步模型训练")