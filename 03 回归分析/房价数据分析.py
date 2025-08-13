import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'sans-serif'  # 使用默认字体
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与清洗
def load_and_explore_data():
    """加载数据并进行初步探索"""
    print("=== 1. 数据加载与探索 ===")
    
    # 读取数据 - 手动设置列名，跳过标题行
    df = pd.read_csv('房价预测数据.csv', skiprows=1, header=None)
    
    # 手动设置正确的列名
    columns = ['房屋ID', '面积', '房间数', '卫生间数', '楼层', '总楼层', 
               '建造年份', '地铁距离', '学校距离', '商场距离', 
               '装修等级', '朝向', '小区类型', '房价']
    df.columns = columns
    
    # 显示数据基本信息
    print(f"数据集形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    
    # 检查数据类型
    print("\n数据类型:")
    print(df.dtypes)
    
    # 检查缺失值
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    
    # 基本信息统计
    print("\n数值型特征统计:")
    numeric_cols = ['面积', '房间数', '卫生间数', '楼层', '总楼层', 
                   '建造年份', '地铁距离', '学校距离', '商场距离', '房价']
    print(df[numeric_cols].describe())
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df.head())
    
    # 检查是否有缺失值
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    
    return df

def feature_engineering(df):
    """特征工程处理"""
    print("\n=== 2. 特征工程 ===")
    
    # 创建副本
    df_processed = df.copy()
    
    # 房龄计算
    current_year = 2025
    df_processed['房龄'] = current_year - df_processed['建造年份']
    
    # 楼层比例
    df_processed['楼层比例'] = df_processed['楼层'] / df_processed['总楼层']
    
    # 房间密度
    df_processed['房间密度'] = df_processed['面积'] / df_processed['房间数']
    
    # 距离总和（交通便利性）
    df_processed['距离总分'] = (df_processed['地铁距离'] + 
                           df_processed['学校距离'] + 
                           df_processed['商场距离'])
    
    # 装修等级数值化
    decoration_map = {'毛坯': 1, '简装修': 2, '精装修': 3, '豪华装修': 4}
    df_processed['装修等级_num'] = df_processed['装修等级'].map(decoration_map)
    
    # 小区类型数值化
    estate_map = {'普通小区': 1, '高档小区': 2, '豪华小区': 3}
    df_processed['小区类型_num'] = df_processed['小区类型'].map(estate_map)
    
    # 朝向独热编码
    df_encoded = pd.get_dummies(df_processed, columns=['朝向'], prefix='朝向')
    
    print("新增特征:")
    new_features = ['房龄', '楼层比例', '房间密度', '距离总分', '装修等级_num', '小区类型_num']
    print(new_features)
    
    return df_encoded

def create_visualizations(df):
    """创建可视化分析"""
    print("\n=== 3. 可视化分析 ===")
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 房价分布
    plt.subplot(3, 3, 1)
    plt.hist(df['房价'], bins=20, color='skyblue', alpha=0.7)
    plt.title('房价分布')
    plt.xlabel('价格 (万元)')
    plt.ylabel('频数')
    
    # 2. 面积与房价关系
    plt.subplot(3, 3, 2)
    plt.scatter(df['面积'], df['房价'], alpha=0.7)
    plt.title('面积与房价关系')
    plt.xlabel('面积 (㎡)')
    plt.ylabel('价格 (万元)')
    
    # 3. 小区类型与房价
    plt.subplot(3, 3, 3)
    sns.boxplot(x='小区类型', y='房价', data=df)
    plt.title('不同小区类型的房价分布')
    
    # 4. 装修等级与房价
    plt.subplot(3, 3, 4)
    sns.boxplot(x='装修等级', y='房价', data=df)
    plt.title('不同装修等级的房价分布')
    plt.xticks(rotation=45)
    
    # 5. 房龄与房价关系
    df_age = df.copy()
    df_age['房龄'] = 2025 - df['建造年份']
    plt.subplot(3, 3, 5)
    plt.scatter(df_age['房龄'], df['房价'], alpha=0.7)
    plt.title('房龄与房价关系')
    plt.xlabel('房龄 (年)')
    plt.ylabel('价格 (万元)')
    
    # 6. 地铁距离与房价
    plt.subplot(3, 3, 6)
    plt.scatter(df['地铁距离'], df['房价'], alpha=0.7)
    plt.title('地铁距离与房价关系')
    plt.xlabel('地铁距离 (km)')
    plt.ylabel('价格 (万元)')
    
    # 7. 相关性热图
    plt.subplot(3, 3, 7)
    numeric_cols = ['面积', '房间数', '卫生间数', '楼层', '总楼层', '建造年份', 
                   '地铁距离', '学校距离', '商场距离', '房价']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('数值型特征相关性热图')
    
    # 8. 价格按房型和小区类型分组
    plt.subplot(3, 3, 8)
    avg_price_by_type = df.groupby(['房间数', '小区类型'])['房价'].mean().unstack()
    avg_price_by_type.plot(kind='bar')
    plt.title('房间数与小区类型的房价均值')
    plt.xlabel('房间数')
    plt.ylabel('平均价格 (万元)')
    plt.legend(title='小区类型')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('房价数据分析可视化.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数执行
if __name__ == "__main__":
    # 加载数据
    df = load_and_explore_data()
    
    # 特征工程
    df_processed = feature_engineering(df)
    
    # 保存处理后的数据
    df_processed.to_csv('房价数据_处理后.csv', index=False, encoding='utf-8-sig')
    print("\n处理后数据已保存至: 房价数据_处理后.csv")
    
    # 创建可视化
    create_visualizations(df)
    
    print("\n数据加载与清洗步骤完成！")