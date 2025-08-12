import pandas as pd

# 读取数据
df = pd.read_csv('E:/datalogic-main/10 增长模型/裂变.csv')

# 打印前几行
print("数据前5行：")
print(df.head())
print("\n")

# 检查裂变类型分布
print("各裂变类型用户数统计：")
print(df['裂变类型'].value_counts())
print("\n")

# 计算各组转化率
print("各裂变类型的转化情况：")
conversion_stats = df.groupby('裂变类型')['是否转化'].agg(['count', 'sum', 'mean'])
print(conversion_stats)
print("\n")

# 保存结果到txt文件
with open('E:/datalogic-main/10 增长模型/01_裂变策略效果评估/basic_stats.txt', 'w', encoding='utf-8') as f:
    f.write("数据前5行：\n")
    f.write(str(df.head()) + "\n\n")
    
    f.write("各裂变类型用户数统计：\n")
    f.write(str(df['裂变类型'].value_counts()) + "\n\n")
    
    f.write("各裂变类型的转化情况：\n")
    f.write(str(conversion_stats) + "\n")

print("基础统计分析完成，结果已保存至 basic_stats.txt")