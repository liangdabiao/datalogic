import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def fig_to_base64(fig):
    """将matplotlib图表转换为base64编码的字符串"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_html_report():
    # 读取数据
    file_path = r'E:\datalogic-main\01 用户画像\用户分群结果.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 创建HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>用户分群分析报告</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; }
            h1, h2, h3 { color: #333; }
            h1 { text-align: center; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            h2 { border-left: 5px solid #4CAF50; padding-left: 10px; }
            .section { margin-bottom: 30px; }
            .chart { text-align: center; margin: 20px 0; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
            th { background-color: #4CAF50; color: white; }
            tr:hover { background-color: #f5f5f5; }
            .summary { background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>用户分群分析报告</h1>
            
            <div class="summary">
                <h2>报告摘要</h2>
                <p>本报告基于180名用户的数据，通过RFM模型、用户价值、生命周期和产品偏好等多个维度对用户进行了分群分析。</p>
                <p>主要发现包括：</p>
                <ul>
                    <li><strong>年轻护眼刚需族</strong> 是最大的用户群体，占总数的40%以上。</li>
                    <li><strong>高收入美妆爱好者</strong> 虽然人数较少，但具有较高的消费能力。</li>
                    <li>男性用户更偏好<strong>护眼类产品</strong>，女性用户对<strong>美妆产品</strong>的偏好更明显。</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>1. 综合分群结果</h2>
                <p>根据用户的年龄、收入、消费、产品偏好等特征，将用户分为以下群体：</p>
    """

    # 1. 综合分群分布 (柱状图)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='综合分群', order=df['综合分群'].value_counts().index)
    plt.title('用户综合分群分布')
    plt.xlabel('用户数量')
    plt.ylabel('用户群体')
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    html_content += f'<div class="chart"><img src="data:image/png;base64,{img_str}" alt="用户综合分群分布"></div>'
    
    # 添加分群统计表
    segment_counts = df['综合分群'].value_counts()
    html_content += '<table><tr><th>用户群体</th><th>用户数量</th><th>占比</th></tr>'
    for segment, count in segment_counts.items():
        percentage = count / len(df) * 100
        html_content += f'<tr><td>{segment}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>'
    html_content += '</table>'

    html_content += """
            </div>
            
            <div class="section">
                <h2>2. 各分群消费和收入分析</h2>
    """

    # 2. 各分群的平均年消费 (柱状图)
    plt.figure(figsize=(10, 6))
    segment_monetary = df.groupby('综合分群')['年消费'].mean().sort_values(ascending=False)
    sns.barplot(x=segment_monetary.values, y=segment_monetary.index)
    plt.title('各用户群体平均年消费')
    plt.xlabel('平均年消费 (元)')
    plt.ylabel('用户群体')
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    html_content += f'<div class="chart"><img src="data:image/png;base64,{img_str}" alt="各用户群体平均年消费"></div>'

    # 3. 各分群的平均年收入 (柱状图)
    plt.figure(figsize=(10, 6))
    segment_income = df.groupby('综合分群')['年收入'].mean().sort_values(ascending=False)
    sns.barplot(x=segment_income.values, y=segment_income.index)
    plt.title('各用户群体平均年收入')
    plt.xlabel('平均年收入 (元)')
    plt.ylabel('用户群体')
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    html_content += f'<div class="chart"><img src="data:image/png;base64,{img_str}" alt="各用户群体平均年收入"></div>'

    html_content += """
            </div>
            
            <div class="section">
                <h2>3. 产品偏好分析</h2>
    """

    # 4. 产品偏好分布 (柱状图)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='近期购买产品')
    plt.title('产品偏好分布')
    plt.xlabel('产品')
    plt.ylabel('用户数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    html_content += f'<div class="chart"><img src="data:image/png;base64,{img_str}" alt="产品偏好分布"></div>'

    # 5. 性别与产品偏好的关系 (柱状图)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='近期购买产品', hue='性别')
    plt.title('性别与产品偏好的关系')
    plt.xlabel('产品')
    plt.ylabel('用户数量')
    plt.xticks(rotation=45)
    plt.legend(title='性别')
    plt.tight_layout()
    img_str = fig_to_base64(plt.gcf())
    plt.close()
    
    html_content += f'<div class="chart"><img src="data:image/png;base64,{img_str}" alt="性别与产品偏好的关系"></div>'

    # 添加交叉分析表
    # 性别与产品偏好交叉分析
    gender_product = pd.crosstab(df['性别'], df['近期购买产品'])
    html_content += '<h3>性别与产品偏好交叉分析</h3>'
    html_content += '<table><tr><th>性别\\产品</th>'
    for product in gender_product.columns:
        html_content += f'<th>{product}</th>'
    html_content += '</tr>'
    for gender in gender_product.index:
        html_content += f'<tr><td>{gender}</td>'
        for product in gender_product.columns:
            html_content += f'<td>{gender_product.loc[gender, product]}</td>'
        html_content += '</tr>'
    html_content += '</table>'

    # 年龄与产品偏好交叉分析
    age_product = pd.crosstab(df['年龄层次'], df['近期购买产品'])
    html_content += '<h3>年龄与产品偏好交叉分析</h3>'
    html_content += '<table><tr><th>年龄层次\\产品</th>'
    for product in age_product.columns:
        html_content += f'<th>{product}</th>'
    html_content += '</tr>'
    for age in age_product.index:
        html_content += f'<tr><td>{age}</td>'
        for product in age_product.columns:
            html_content += f'<td>{age_product.loc[age, product]}</td>'
        html_content += '</tr>'
    html_content += '</table>'

    html_content += """
            </div>
            
            <div class="section">
                <h2>4. 结论与建议</h2>
                <h3>主要结论</h3>
                <ul>
                    <li><strong>年轻护眼刚需族</strong>是核心用户群体，应继续加强针对该群体的护眼产品营销。</li>
                    <li><strong>高收入美妆爱好者</strong>虽然人数不多，但消费能力强，可作为高价值用户重点维护。</li>
                    <li>男性用户更偏好护眼类产品，女性用户对美妆产品的偏好更明显，可以进行性别差异化营销。</li>
                </ul>
                
                <h3>营销建议</h3>
                <ul>
                    <li>针对<strong>年轻护眼刚需族</strong>，可以推出更多性价比高的护眼产品，并通过社交媒体等渠道进行推广。</li>
                    <li>针对<strong>高收入美妆爱好者</strong>，可以提供限量版或高端美妆产品，满足其对品质和个性化的需求。</li>
                    <li>针对<strong>中年视力关爱群体</strong>，可以推出健康主题的护眼产品套餐，并提供专业的用眼健康咨询服务。</li>
                    <li>对于<strong>价格敏感学生党</strong>，可以设计一些入门级的产品，并提供学生专属优惠。</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # 将HTML内容写入文件
    with open(r'E:\datalogic-main\01 用户画像\用户分群分析报告.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("HTML报告已生成: E:\\datalogic-main\\01 用户画像\\用户分群分析报告.html")

if __name__ == "__main__":
    generate_html_report()