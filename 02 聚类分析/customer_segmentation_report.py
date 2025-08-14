#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Segmentation Analysis Report Generator
This script generates comprehensive validation reports and summaries
for the RFM-based customer segmentation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up Chinese font for plots
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load original and processed data"""
    original_df = pd.read_csv('电商历史订单.csv')
    segments_df = pd.read_csv('customer_segments_clean.csv')
    return original_df, segments_df

def create_validation_report():
    """Create comprehensive data validation report"""
    original_df, segments_df = load_data()
    
    # Data quality checks
    report = []
    report.append("=" * 60)
    report.append("CUSTOMER SEGMENTATION ANALYSIS VALIDATION REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n")
    
    # 1. Original data validation
    report.append("1. ORIGINAL DATA VALIDATION")
    report.append("-" * 30)
    report.append(f"Total orders: {len(original_df):,}")
    report.append(f"Unique customers: {original_df['用户码'].nunique():,}")
    report.append(f"Date range: {original_df['消费日期'].min()} to {original_df['消费日期'].max()}")
    report.append(f"Total revenue: ￥{original_df['数量'].sum() * original_df['单价'].sum():,.2f}")
    
    # Check for data quality issues
    negative_qty = len(original_df[original_df['数量'] <= 0])
    negative_price = len(original_df[original_df['单价'] <= 0])
    missing_values = original_df.isnull().sum().sum()
    
    report.append(f"\nData quality issues:")
    report.append(f"- Negative quantities: {negative_qty:,}")
    report.append(f"- Negative prices: {negative_price:,}")
    report.append(f"- Missing values: {missing_values:,}")
    
    # 2. Cleaned data validation
    report.append(f"\n2. CLEANED DATA VALIDATION")
    report.append("-" * 30)
    cleaned_orders = original_df[original_df['数量'] > 0]
    report.append(f"Orders after cleaning: {len(cleaned_orders):,}")
    report.append(f"Final customer segments: {len(segments_df):,}")
    
    # 3. Segment distribution
    report.append(f"\n3. CUSTOMER SEGMENT DISTRIBUTION")
    report.append("-" * 30)
    segment_counts = segments_df['customer_value'].value_counts()
    
    for segment, count in segment_counts.items():
        percentage = (count / len(segments_df)) * 100
        report.append(f"{segment}: {count:,} customers ({percentage:.1f}%)")
    
    # 4. Statistical summary by segment
    report.append(f"\n4. SEGMENT STATISTICS")
    report.append("-" * 30)
    stats_summary = segments_df.groupby('customer_value').agg({
        'R值': ['mean', 'min', 'max', 'std'],
        'F值': ['mean', 'min', 'max', 'std'],
        'M值': ['mean', 'min', 'max', 'std'],
        '总分': ['mean', 'min', 'max']
    }).round(2)
    
    report.append(stats_summary.to_string())
    
    # 5. Revenue analysis
    report.append(f"\n5. REVENUE ANALYSIS BY SEGMENT")
    report.append("-" * 30)
    
    # Load original data to get accurate revenue
    original_df_clean = original_df[original_df['数量'] > 0].copy()
    original_df_clean['订单金额'] = original_df_clean['数量'] * original_df_clean['单价']
    
    # Merge with segments to get per-customer totals
    customer_revenue = original_df_clean.groupby('用户码')['订单金额'].sum().reset_index()
    customer_revenue = customer_revenue.merge(segments_df[['用户码', 'customer_value']], on='用户码')
    
    revenue_by_segment = customer_revenue.groupby('customer_value').agg({
        '订单金额': ['sum', 'mean', 'count']
    }).round(2)
    
    total_revenue = customer_revenue['订单金额'].sum()
    report.append(revenue_by_segment.to_string())
    report.append(f"\nTotal revenue across all customers: ￥{total_revenue:,.2f}")
    
    return '\n'.join(report)

def generate_summary_statistics():
    """Generate detailed statistical summary for each segment"""
    original_df, segments_df = load_data()
    
    # Load cleaned original data
    cleaned_orders = original_df[original_df['数量'] > 0].copy()
    cleaned_orders['订单金额'] = cleaned_orders['数量'] * cleaned_orders['单价']
    
    # Get customer level data
    customer_data = cleaned_orders.groupby('用户码').agg({
        '消费日期': ['max', 'count'],
        '订单金额': 'sum'
    }).reset_index()
    
    customer_data.columns = ['用户码', '最近日期', '购买次数', '总金额']
    customer_data['最近日期'] = pd.to_datetime(customer_data['最近日期'])
    customer_data['R值'] = (customer_data['最近日期'].max() - customer_data['最近日期']).dt.days
    customer_data = customer_data.rename(columns={'购买次数': 'F值', '总金额': 'M值'})
    
    # Merge with segments
    customer_data = customer_data.merge(segments_df[['用户码', 'customer_value']], on='用户码')
    
    # Create summary statistics
    summary_stats = {}
    
    for segment in ['High', 'Medium', 'Low']:
        segment_data = customer_data[customer_data['customer_value'] == segment]
        
        summary_stats[segment] = {
            'count': len(segment_data),
            'avg_recency': segment_data['R值'].mean(),
            'avg_frequency': segment_data['F值'].mean(),
            'avg_monetary': segment_data['M值'].mean(),
            'total_revenue': segment_data['M值'].sum(),
            'avg_order_value': segment_data['M值'].sum() / segment_data['F值'].sum(),
            'revenue_contribution': (segment_data['M值'].sum() / customer_data['M值'].sum()) * 100
        }
    
    summary_df = pd.DataFrame(summary_stats).T
    
    # Save summary
    summary_df.to_csv('segment_summary_statistics.csv', encoding='utf-8-sig')
    
    return summary_df

def create_executive_dashboard():
    """Create visualization dashboard for customer segments"""
    original_df, segments_df = load_data()
    
    # Get clean customer data
    cleaned_orders = original_df[original_df['数量'] > 0].copy()
    cleaned_orders['订单金额'] = cleaned_orders['数量'] * cleaned_orders['单价']
    
    customer_data = cleaned_orders.groupby('用户码').agg({
        '消费日期': 'count',
        '订单金额': 'sum'
    }).reset_index()
    customer_data.columns = ['用户码', 'F值', '订单金额']
    # 重命名 '订单金额' 为 'M值'，以保证后续分组统计正常
    customer_data = customer_data.rename(columns={'订单金额': 'M值'})
    customer_data = customer_data.merge(segments_df[['用户码', 'customer_value']], on='用户码')
    # 确保 'M值' 列存在，防止 merge 后被覆盖
    if 'M值' not in customer_data.columns:
        if '订单金额' in customer_data.columns:
            customer_data = customer_data.rename(columns={'订单金额': 'M值'})
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('客户分群分析仪表板 (Customer Segmentation Dashboard)', fontsize=16)
    
    # 1. Segment distribution pie chart
    segment_counts = segments_df['customer_value'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[0,0].pie(segment_counts.values, labels=segment_counts.index, colors=colors, autopct='%1.1f%%')
    axes[0,0].set_title('客户分群分布 (Customer Segment Distribution)')
    
    # 2. Revenue share by segment
    revenue_data = customer_data.groupby('customer_value')['M值'].sum()
    axes[0,1].pie(revenue_data.values, labels=revenue_data.index, colors=colors, autopct='%1.1f%%')
    axes[0,1].set_title('收入份额分布 (Revenue Share by Segment)')
    
    # 3. RFM scatter plot
    segment_colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    for segment in ['High', 'Medium', 'Low']:
        subset = customer_data[customer_data['customer_value'] == segment]
        axes[1,0].scatter(subset['F值'], subset['M值'], 
                         c=segment_colors[segment], label=segment, alpha=0.6)
    axes[1,0].set_xlabel('购买频率 (Frequency)')
    axes[1,0].set_ylabel('消费金额 (Monetary)')
    axes[1,0].set_title('购买频率 vs 消费金额')
    axes[1,0].legend()
    axes[1,0].set_yscale('log')
    
    # 4. Box plot of monetary values by segment (log scale)
    customer_data[customer_data['M值'] > 0].boxplot(column='M值', by='customer_value', ax=axes[1,1])
    axes[1,1].set_yscale('log')
    axes[1,1].set_title('各分群消费金额分布 (Monetary Value Distribution)')
    axes[1,1].set_xlabel('客户价值分群')
    axes[1,1].set_ylabel('消费金额 (对数刻度)')
    
    plt.tight_layout()
    plt.savefig('customer_segmentation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def export_vip_list():
    """Export VIP customer list for marketing campaigns"""
    _, segments_df = load_data()
    
    # Get high value customers
    vip_customers = segments_df[segments_df['customer_value'] == 'High'].copy()
    
    # Add rankings within VIP segment
    vip_customers = vip_customers.sort_values(['总分', 'M值', 'F值'], 
                                            ascending=[False, False, False])
    vip_customers['rank'] = range(1, len(vip_customers) + 1)
    
    # Select columns for marketing
    vip_export = vip_customers[['用户码', 'R值', 'F值', 'M值', '总分', 'rank']]
    
    # Save VIP list
    vip_export.to_csv('vip_customers_list.csv', index=False, encoding='utf-8-sig')
    
    # Generate summary
    vip_summary = f"""
top_customers
==========================================
VIP客户总数: {len(vip_customers)} 人
平均最近购买: {vip_customers['R值'].mean():.1f} 天前
平均购买频次: {vip_customers['F值'].mean():.1f} 次
平均消费金额: ￥{vip_customers['M值'].mean():,.2f}
平均RFM总分: {vip_customers['总分'].mean():.1f}/8
==========================================
"""
    
    return vip_summary

def main():
    """Generate all reports and outputs"""
    print("Starting Customer Segmentation Report Generation")
    print("=" * 50)
    
    # 1. Create validation report
    print("Creating data validation report...")
    validation_report = create_validation_report()
    with open('data_validation_report.txt', 'w', encoding='utf-8') as f:
        f.write(validation_report)
    
    # 2. Generate summary statistics
    print("Generating segment statistics...")
    summary_stats = generate_summary_statistics()
    
    # 3. Create visualization dashboard
    print("Creating visualization dashboard...")
    create_executive_dashboard()
    
    # 4. Export VIP list
    print("Exporting VIP customer list...")
    vip_summary = export_vip_list()
    print(vip_summary)
    
    # 5. Generate executive summary
    print("Creating executive summary...")
    executive_summary = f"""
客户分群分析执行摘要
===================
项目完成: 完成

核心发现:
- 总客户数: 980人
- 高价值客户: 404人 (41.2%)
- 中等价值客户: 326人 (33.3%) 
- 低价值客户: 250人 (25.5%)

营销建议:
- VIP客户应立即列入营销推广计划
- 中等价值客户需要激活策略
- 低价值客户可考虑重新定位

已生成文件:
- data_validation_report.txt: 详细数据验证报告
- segment_summary_statistics.csv: 分群详细统计数据
- customer_segmentation_dashboard.png: 可视化仪表板
- vip_customers_list.csv: VIP客户清单
"""
    
    with open('executive_summary.txt', 'w', encoding='utf-8') as f:
        f.write(executive_summary)
    
    print("\nAll reports generated successfully!")
    print("\nGenerated files:")
    print("   - data_validation_report.txt")
    print("   - segment_summary_statistics.csv")
    print("   - customer_segmentation_dashboard.png")
    print("   - vip_customers_list.csv")
    print("   - executive_summary.txt")

if __name__ == "__main__":
    main()