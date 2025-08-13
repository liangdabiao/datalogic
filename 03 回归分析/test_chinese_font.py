import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 打印可用字体
print("可用的中文字体:")
for font in fm.fontManager.ttflist:
    if any(keyword in font.name for keyword in ['Sim', 'Hei', 'Song', 'Kai', 'Microsoft']):
        print(f"- {font.name} ({font.fname})")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('房价数据_分析版.csv', encoding='utf-8-sig')

# 创建数据单位转换
df['价格万'] = df['房价'] / 10000
df['房龄'] = 2025 - df['建造年份']

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['价格万'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
ax.set_title('房价分布直方图', fontsize=16)
ax.set_xlabel('价格 (万元)', fontsize=12)
ax.set_ylabel('房源数量', fontsize=12)

# 保存图表
plt.tight_layout()
plt.savefig('test_chinese_font.png', dpi=300, bbox_inches='tight')
plt.show()

print("测试图表已生成: test_chinese_font.png")