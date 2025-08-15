import pandas as pd
import re
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体以支持图表显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据加载与清洗
print("正在加载数据...")
df = pd.read_csv('article.csv')

# 清洗 reply 和 view 列，将 'k' 转换为数字
def convert_count(count_str):
    if pd.isna(count_str):
        return 0
    count_str = str(count_str)
    if 'k' in count_str:
        return float(count_str.replace('k', '')) * 1000
    elif 'm' in count_str: # 防止未来有m
        return float(count_str.replace('m', '')) * 1000000
    else:
        try:
            return float(count_str)
        except ValueError:
            return 0 # 如果转换失败，默认为0

df['reply_num'] = df['reply'].apply(convert_count)
df['view_num'] = df['view'].apply(convert_count)
df['interaction_rate'] = df['reply_num'] / (df['view_num'] + 1) # 加1防止除以0

print("数据加载与清洗完成。")
print(df.head())

# 2. 核心分析

# 2.1 高热度帖子特征分析
print("\n--- 高热度帖子特征分析 ---")
# 定义高热度标准 (Top 10%)
top_reply_threshold = df['reply_num'].quantile(0.90)
top_view_threshold = df['view_num'].quantile(0.90)

df_top_reply = df[df['reply_num'] >= top_reply_threshold]
df_top_view = df[df['view_num'] >= top_view_threshold]

print(f"高回复量帖子阈值 (Top 10%): {top_reply_threshold:.0f} 回复")
print(f"高阅读量帖子阈值 (Top 10%): {top_view_threshold:.0f} 阅读")

# 分析高热度帖子的分类和标签分布
def analyze_distribution(dataframe, name):
    print(f"\n--- {name} 分布 ---")
    # 分类
    category_dist = dataframe['category'].value_counts().head(10)
    print("Top 10 分类:")
    print(category_dist)
    
    # 主标签
    tag1_dist = dataframe['tag-1'].value_counts().dropna().head(10)
    print("\nTop 10 主标签:")
    print(tag1_dist)
    
    # 所有标签
    all_tags = pd.concat([dataframe['tag-1'], dataframe['tag-2'], dataframe['tag-3'], dataframe['tag-4']])
    all_tags_cleaned = all_tags.dropna()
    all_tags_dist = all_tags_cleaned.value_counts().head(10)
    print("\nTop 10 所有标签 (包括次要标签):")
    print(all_tags_dist)

analyze_distribution(df_top_reply, "高回复量帖子")
analyze_distribution(df_top_view, "高阅读量帖子")

# 标题关键词分析 (以高回复量为例)
print("\n--- 高回复量帖子标题关键词分析 ---")
# 合并所有标题
titles_top_reply = ' '.join(df_top_reply['title'].dropna().astype(str))
# 使用jieba分词
words = jieba.lcut(titles_top_reply)
# 过滤掉长度小于2的词和一些无意义的词
stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
# 统计词频
word_counts = Counter(filtered_words)
print("高回复量帖子标题中Top 20关键词:")
print(word_counts.most_common(20))


# 2.2 分类与标签影响力评估
print("\n--- 分类与标签影响力评估 ---")

def evaluate_influence(column_name, df_source):
    print(f"\n--- 按 '{column_name}' 分组的平均互动量 ---")
    # 过滤掉空值
    df_filtered = df_source.dropna(subset=[column_name])
    if df_filtered.empty:
        print(f"列 '{column_name}' 没有有效数据。")
        return
    
    # 计算平均值并排序
    avg_stats = df_filtered.groupby(column_name)[['reply_num', 'view_num']].mean().sort_values(by='reply_num', ascending=False)
    print(avg_stats.head(10))
    
    # 可视化 - 仅对前N个进行可视化
    top_n = 15
    top_categories = avg_stats.head(top_n)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    sns.barplot(ax=axes[0], x=top_categories.index, y=top_categories['reply_num'], palette="Blues_d")
    axes[0].set_title(f'Top {top_n} {column_name} 平均回复数')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('平均回复数')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(ax=axes[1], x=top_categories.index, y=top_categories['view_num'], palette="Greens_d")
    axes[1].set_title(f'Top {top_n} {column_name} 平均阅读数')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('平均阅读数')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{column_name}_avg_interaction.png')
    plt.close() # 关闭图形以释放内存
    print(f"图表已保存为 '{column_name}_avg_interaction.png'")

# 评估 category
evaluate_influence('category', df)

# 评估 tag-1
evaluate_influence('tag-1', df)


# 2.3 内容类型与互动关系 (简化版)
print("\n--- 内容类型与互动关系 ---")
# 基于标题和标签的关键词进行简单分类
def classify_content(row):
    title = str(row['title']).lower()
    tags = [str(row[f'tag-{i}']).lower() for i in range(1, 5)]
    tags_str = ' '.join(tags)

    # 检查关键词
    if '教程' in title or '教程' in tags_str:
        return '教程类'
    elif '求助' in title or '问' in title or '求' in title or '求助' in tags_str or '问' in tags_str or '求' in tags_str:
        return '求助类'
    elif '抽奖' in title or '抽奖' in tags_str:
        return '抽奖/福利类'
    elif 'aff' in title or 'aff' in tags_str:
        return 'AFF推广类'
    elif '新闻' in title or '快讯' in row['category'] or '新闻' in tags_str:
        return '资讯类'
    elif '分享' in title or '分享' in tags_str:
        return '分享类'
    else:
        return '其他/讨论类'

df['content_type'] = df.apply(classify_content, axis=1)

# 计算不同类型内容的平均互动量和互动率
content_type_stats = df.groupby('content_type').agg({
    'reply_num': 'mean',
    'view_num': 'mean',
    'interaction_rate': 'mean'
}).sort_values(by='reply_num', ascending=False)

print("各内容类型的平均互动量和互动率:")
print(content_type_stats)

# 可视化内容类型分析
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

sns.barplot(ax=axes[0], x=content_type_stats.index, y=content_type_stats['reply_num'], palette="Blues_d")
axes[0].set_title('各内容类型平均回复数')
axes[0].set_xlabel('内容类型')
axes[0].set_ylabel('平均回复数')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(ax=axes[1], x=content_type_stats.index, y=content_type_stats['view_num'], palette="Greens_d")
axes[1].set_title('各内容类型平均阅读数')
axes[1].set_xlabel('内容类型')
axes[1].set_ylabel('平均阅读数')
axes[1].tick_params(axis='x', rotation=45)

sns.barplot(ax=axes[2], x=content_type_stats.index, y=content_type_stats['interaction_rate'], palette="Reds_d")
axes[2].set_title('各内容类型平均互动率 (回复数/阅读数)')
axes[2].set_xlabel('内容类型')
axes[2].set_ylabel('平均互动率')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('content_type_interaction.png')
plt.close()
print("内容类型分析图表已保存为 'content_type_interaction.png'")


# 2.4 社区话题趋势分析 (词频)
print("\n--- 社区话题趋势分析 ---")
# 合并所有文本 (标题 + 标签)
all_text = ' '.join(df['title'].dropna().astype(str))
for i in range(1, 5):
    all_text += ' ' + ' '.join(df[f'tag-{i}'].dropna().astype(str))

# 分词
all_words = jieba.lcut(all_text)
# 过滤
filtered_all_words = [word for word in all_words if len(word) > 1 and word not in stop_words]
# 统计
all_word_counts = Counter(filtered_all_words)

print("整个社区Top 30高频词:")
print(all_word_counts.most_common(30))

# 可视化词云 (需要安装 wordcloud 库，这里先用条形图)
top_words = all_word_counts.most_common(20)
words_list, counts_list = zip(*top_words) # 解包

plt.figure(figsize=(12, 8))
sns.barplot(x=list(counts_list), y=list(words_list), palette="viridis")
plt.title('社区Top 20高频词')
plt.xlabel('频次')
plt.ylabel('词语')
plt.tight_layout()
plt.savefig('community_word_frequency.png')
plt.close()
print("社区词频分析图表已保存为 'community_word_frequency.png'")


# 2.6 用户行为洞察 (互动率)
print("\n--- 用户行为洞察 (互动率) ---")
# 找出互动率最高和最低的帖子 (排除view_num为0的)
df_for_rate = df[df['view_num'] > 0]

if not df_for_rate.empty:
    top_rate_posts = df_for_rate.nlargest(5, 'interaction_rate')[['title', 'category', 'reply_num', 'view_num', 'interaction_rate']]
    bottom_rate_posts = df_for_rate.nsmallest(5, 'interaction_rate')[['title', 'category', 'reply_num', 'view_num', 'interaction_rate']]

    print("\n互动率最高的5个帖子:")
    print(top_rate_posts.to_string(index=False))
    
    print("\n互动率最低的5个帖子:")
    print(bottom_rate_posts.to_string(index=False))
else:
    print("没有足够的数据来计算互动率。")

print("\n数据分析完成。")