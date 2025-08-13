"""
LLM 版本内容分析脚本
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import concurrent.futures
from http import HTTPStatus

# --- LLM Analysis Specific Imports ---
import dashscope
from dashscope import Generation
# ------------------------------------

def main():
    # 0. 设置绘图字体
    plt.rcParams["font.family"] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 设定API密钥
    # 注意：不要将API密钥直接写入代码，应使用环境变量
    # 请在运行此脚本前设置环境变量 DASHSCOPE_API_KEY
    dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
    if not dashscope.api_key:
        print("警告: 未设置环境变量 DASHSCOPE_API_KEY。将跳过LLM分析部分。")
        # 为了演示，我们生成一些模拟数据
        # raise ValueError("请设置环境变量 DASHSCOPE_API_KEY。例如，在命令行运行: set DASHSCOPE_API_KEY=your_actual_api_key_here (Windows) 或 export DASHSCOPE_API_KEY=your_actual_api_key_here (Linux/Mac)")
        df_llm_results = generate_mock_data()
        visualize_results(df_llm_results)
        return

    print("开始载入数据...")
    # 2. 载入数据集
    df_video = pd.read_csv("分类视频.csv")
    print(f"数据载入完成，共 {len(df_video)} 条记录。")

    # 3. 数据预处理
    # 只保留需要分析的列
    df_llm = df_video[['分类', '标签', '说明文字']].copy()
    # 去除空值
    df_llm.dropna(subset=['标签', '说明文字'], inplace=True)
    # 重置索引
    df_llm.reset_index(drop=True, inplace=True)
    print(f"数据预处理完成，剩余 {len(df_llm)} 条有效记录。")

    # 4. 采样数据以降低成本（示例：每个分类取前5个非空记录）
    sample_size_per_category = 5
    df_sampled = df_llm.groupby('分类').apply(lambda x: x.head(sample_size_per_category)).reset_index(drop=True)
    print(f"数据采样完成，采样 {len(df_sampled)} 条记录用于LLM分析。")

    # 5. 执行LLM分析
    print("开始调用LLM进行分析...")
    llm_results = perform_llm_analysis(df_sampled)
    print("LLM分析完成。")

    # 6. 将结果存入DataFrame
    df_llm_results = pd.DataFrame(llm_results)
    
    # 7. 解析LLM输出
    print("解析LLM输出...")
    df_llm_results = parse_llm_output(df_llm_results)
    print("LLM输出解析完成。")

    # 8. 展示部分结果
    print("\n--- LLM 分析结果 (采样数据) ---")
    print(df_llm_results[['分类', '标签', '标签情绪类别', '说明情绪类别', '主题']].head(10))

    # 9. 可视化
    visualize_results(df_llm_results)

def call_qwen(prompt, model='qwen-plus'):
    """调用通义千问API"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model=model,
                prompt=prompt,
                seed=1234,
                max_tokens=200, # 限制输出长度
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                print(f'请求失败 (尝试 {attempt+1}/{max_retries}): {response.status_code} - {response.message}')
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # 指数退避
        except Exception as e:
            print(f'调用API时出错 (尝试 {attempt+1}/{max_retries}): {e}')
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None

def get_sentiment_prompt(text):
    """生成情绪分析的Prompt"""
    return f"请分析以下文本的情绪倾向，只需要回答情绪类别（如：非常积极, 积极, 中性, 消极, 非常消极）和一个0到1之间的置信度分数，用英文逗号分隔。文本：{text}"

def get_topic_prompt(text):
    """生成主题提取的Prompt"""
    return f"请从以下文本中提取最多3个核心主题关键词或短语，用英文逗号分隔。文本：{text}"

def process_row(row):
    """处理单行数据"""
    category = row['分类']
    tags = row['标签']
    description = row['说明文字']
    
    # 情绪分析 (对标签和说明文字分别进行)
    sentiment_tags_result = call_qwen(get_sentiment_prompt(tags))
    time.sleep(0.5) # 简单的速率限制
    sentiment_desc_result = call_qwen(get_sentiment_prompt(description))
    time.sleep(0.5)
    
    # 主题提取 (对说明文字进行)
    topics_result = call_qwen(get_topic_prompt(description))
    time.sleep(0.5)
    
    return {
        '分类': category,
        '标签': tags,
        '说明文字': description,
        '标签情绪': sentiment_tags_result,
        '说明情绪': sentiment_desc_result,
        '主题': topics_result
    }

def perform_llm_analysis(df_sampled):
    """执行LLM分析"""
    llm_results = []
    # 使用线程池并行处理 (注意API可能有并发限制)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_row, row): row for _, row in df_sampled.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            row = futures[future]
            try:
                result = future.result()
                if result:
                    llm_results.append(result)
                    print(f"进度: {i+1}/{len(futures)} - 已处理分类 '{row['分类']}'")
            except Exception as e:
                print(f"处理行时出错: {e}")
                # 可以选择添加一个包含错误信息的占位结果
                llm_results.append({
                    '分类': row['分类'],
                    '标签': row['标签'],
                    '说明文字': row['说明文字'],
                    '标签情绪': 'Error',
                    '说明情绪': 'Error',
                    '主题': 'Error'
                })
    return llm_results

def parse_llm_output(df_llm_results):
    """解析LLM输出"""
    def parse_sentiment(sentiment_str):
        if not sentiment_str or sentiment_str == 'Error':
            return pd.Series([None, None])
        parts = sentiment_str.split(',')
        if len(parts) >= 2:
            sentiment_label = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                confidence = None
            return pd.Series([sentiment_label, confidence])
        else:
            return pd.Series([sentiment_str, None])

    df_llm_results[['标签情绪类别', '标签情绪置信度']] = df_llm_results['标签情绪'].apply(parse_sentiment)
    df_llm_results[['说明情绪类别', '说明情绪置信度']] = df_llm_results['说明情绪'].apply(parse_sentiment)
    
    # 主题解析可以更复杂，这里只做简单分割
    return df_llm_results

def visualize_results(df_llm_results):
    """可视化结果"""
    print("\n开始生成可视化图表...")
    
    # 1. 情绪分布 (标签)
    plt.figure(figsize=(10, 6))
    df_valid_sentiment_tags = df_llm_results.dropna(subset=['标签情绪类别'])
    if not df_valid_sentiment_tags.empty:
        # 定义情绪类别的顺序
        sentiment_order = ['非常消极', '消极', '中性', '积极', '非常积极']
        # 统计每个类别的情绪分布
        sentiment_counts = df_valid_sentiment_tags['标签情绪类别'].value_counts().reindex(sentiment_order, fill_value=0)
        
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')
        plt.title("LLM分析的标签情绪分布", fontsize=15)
        plt.xlabel("情绪类别", fontsize=14)
        plt.ylabel("数目", fontsize=14)
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数字标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("情绪分布_标签_LLM.png")
        plt.show()
        print("图表 '情绪分布_标签_LLM.png' 已保存。")
    else:
        print("标签情绪分析结果不足，无法生成情绪分布图表。")

    # 2. 不同分类的情绪对比 (说明文字)
    plt.figure(figsize=(12, 6))
    df_valid_sentiment_desc = df_llm_results.dropna(subset=['说明情绪类别', '分类'])
    if not df_valid_sentiment_desc.empty:
        # 获取所有唯一分类和情绪类别
        categories = sorted(df_valid_sentiment_desc['分类'].unique())
        sentiments = ['非常消极', '消极', '中性', '积极', '非常积极']
        
        # 创建一个空的计数字典
        heatmap_data = {cat: {s: 0 for s in sentiments} for cat in categories}
        
        # 填充计数字典
        for _, row in df_valid_sentiment_desc.iterrows():
            cat = row['分类']
            sent = row['说明情绪类别']
            if sent in sentiments: # 确保情绪类别是预期的
                 heatmap_data[cat][sent] += 1

        # 转换为 DataFrame 用于绘图
        heatmap_df = pd.DataFrame(heatmap_data).T # .T 是转置
        heatmap_df = heatmap_df[sentiments] # 确保列顺序
        
        # 绘制热力图
        sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("各分类的说明文字情绪分布 (热力图)", fontsize=15)
        plt.xlabel("情绪类别", fontsize=14)
        plt.ylabel("视频分类", fontsize=14)
        plt.tight_layout()
        plt.savefig("情绪分布_分类_说明_LLM.png")
        plt.show()
        print("图表 '情绪分布_分类_说明_LLM.png' 已保存。")
    else:
        print("说明文字情绪分析结果不足，无法生成分类情绪对比图表。")

    # 3. 主题词云 (示例: 简单的词频统计)
    from collections import Counter
    all_topics = []
    for topics_str in df_llm_results['主题'].dropna():
        if topics_str != 'Error':
            topics_list = [t.strip() for t in topics_str.split(',')]
            all_topics.extend(topics_list)
    
    if all_topics:
        topic_counts = Counter(all_topics)
        # 取前N个最常见的主题
        top_n = 20
        top_topics = dict(topic_counts.most_common(top_n))
        
        plt.figure(figsize=(10, 6))
        topics_sorted = sorted(top_topics.items(), key=lambda item: item[1])
        topic_names, counts = zip(*topics_sorted)
        
        bars = plt.barh(topic_names, counts, color='lightcoral')
        plt.title(f"LLM提取的热门主题词 Top {top_n}", fontsize=15)
        plt.xlabel("出现次数", fontsize=14)
        plt.ylabel("主题词", fontsize=14)
        plt.tight_layout()
        plt.savefig("热门主题词_LLM.png")
        plt.show()
        print("图表 '热门主题词_LLM.png' 已保存。")
    else:
        print("主题提取结果不足，无法生成主题词云图。")

    print("可视化完成。")

# --- 模拟数据生成函数 ---
def generate_mock_data():
    """当没有API密钥时，生成模拟数据用于演示"""
    print("生成模拟数据用于演示...")
    np.random.seed(42)
    categories = ['科技', '娱乐', '新闻', '游戏', '动画', '博客', '宠物']
    sentiments = ['非常积极', '积极', '中性', '消极', '非常消极']
    
    mock_data = []
    for i in range(35): # 5个分类 * 7个样本
        cat = np.random.choice(categories)
        tag_sent = np.random.choice(sentiments)
        desc_sent = np.random.choice(sentiments)
        topics = ', '.join(np.random.choice(['技术', '创新', '幽默', '信息', '挑战', '故事', '教程', '评测'], size=3, replace=False))
        
        mock_data.append({
            '分类': cat,
            '标签': f"标签{i}",
            '说明文字': f"说明文字{i}",
            '标签情绪类别': tag_sent,
            '标签情绪置信度': round(np.random.rand(), 2),
            '说明情绪类别': desc_sent,
            '说明情绪置信度': round(np.random.rand(), 2),
            '主题': topics
        })
    
    return pd.DataFrame(mock_data)
# -------------------------

if __name__ == "__main__":
    main()
    print("\nLLM版本内容分析脚本执行完毕。")