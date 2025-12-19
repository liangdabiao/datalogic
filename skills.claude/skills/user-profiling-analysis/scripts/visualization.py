#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户画像可视化脚本
生成分群分析图表和可视化报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UserProfilerVisualizer:
    """用户画像可视化器"""

    def __init__(self, data_path, output_dir="visualization_output"):
        """
        初始化可视化器

        Args:
            data_path: 分析结果数据文件路径
            output_dir: 图表输出目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None

    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"✅ 成功加载数据: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def plot_segment_distribution(self):
        """绘制用户分群分布图"""
        if '综合分群' not in self.df.columns:
            print("❌ 数据中缺少'综合分群'列")
            return

        plt.figure(figsize=(12, 8))
        segment_counts = self.df['综合分群'].value_counts()

        # 创建横向条形图
        bars = plt.barh(range(len(segment_counts)), segment_counts.values)

        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # 添加数据标签
        for i, (segment, count) in enumerate(segment_counts.items()):
            plt.text(count + 0.5, i, str(count), va='center')

        plt.yticks(range(len(segment_counts)), segment_counts.index)
        plt.xlabel('用户数量')
        plt.title('用户画像分群分布', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / "用户分群分布.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def plot_consumption_by_segment(self):
        """绘制各分群平均消费对比图"""
        if '综合分群' not in self.df.columns or '年消费' not in self.df.columns:
            print("❌ 数据中缺少必要的列")
            return

        plt.figure(figsize=(12, 8))
        segment_consumption = self.df.groupby('综合分群')['年消费'].mean().sort_values(ascending=False)

        bars = plt.bar(range(len(segment_consumption)), segment_consumption.values)
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_consumption)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # 添加数据标签
        for i, value in enumerate(segment_consumption.values):
            plt.text(i, value + 2, f'{value:.1f}元', ha='center', va='bottom')

        plt.xticks(range(len(segment_consumption)), segment_consumption.index, rotation=45, ha='right')
        plt.ylabel('平均年消费 (元)')
        plt.title('各用户群体平均年消费对比', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / "各分群平均消费.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def plot_income_by_segment(self):
        """绘制各分群平均收入对比图"""
        if '综合分群' not in self.df.columns or '年收入' not in self.df.columns:
            print("❌ 数据中缺少必要的列")
            return

        plt.figure(figsize=(12, 8))
        segment_income = self.df.groupby('综合分群')['年收入'].mean().sort_values(ascending=False)

        bars = plt.bar(range(len(segment_income)), segment_income.values / 10000)  # 转换为万元
        colors = plt.cm.plasma(np.linspace(0, 1, len(segment_income)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # 添加数据标签
        for i, value in enumerate(segment_income.values):
            plt.text(i, value/10000 + 0.2, f'{value/10000:.1f}万', ha='center', va='bottom')

        plt.xticks(range(len(segment_income)), segment_income.index, rotation=45, ha='right')
        plt.ylabel('平均年收入 (万元)')
        plt.title('各用户群体平均年收入对比', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / "各分群平均收入.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def plot_demographic_distribution(self):
        """绘制人口统计学分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('用户人口统计学特征分布', fontsize=20, fontweight='bold')

        # 性别分布
        if '性别' in self.df.columns:
            gender_counts = self.df['性别'].value_counts()
            axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('性别分布')

        # 年龄分布
        if '年龄' in self.df.columns:
            axes[0, 1].hist(self.df['年龄'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('年龄分布')
            axes[0, 1].set_xlabel('年龄')
            axes[0, 1].set_ylabel('人数')

        # 收入分布
        if '年收入' in self.df.columns:
            axes[1, 0].hist(self.df['年收入'] / 10000, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('年收入分布')
            axes[1, 0].set_xlabel('年收入 (万元)')
            axes[1, 0].set_ylabel('人数')

        # 年龄层次分布
        if '年龄层次' in self.df.columns:
            age_group_counts = self.df['年龄层次'].value_counts()
            bars = axes[1, 1].bar(range(len(age_group_counts)), age_group_counts.values)
            colors = plt.cm.Set2(np.linspace(0, 1, len(age_group_counts)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            axes[1, 1].set_xticks(range(len(age_group_counts)))
            axes[1, 1].set_xticklabels(age_group_counts.index, rotation=45, ha='right')
            axes[1, 1].set_title('年龄层次分布')

        plt.tight_layout()
        output_file = self.output_dir / "人口统计学分布.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def plot_rfm_analysis(self):
        """绘制RFM分析图表"""
        if not all(col in self.df.columns for col in ['R_Stage', 'F_Stage', 'M_Stage']):
            print("❌ 数据中缺少RFM分析结果列")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('RFM分析结果', fontsize=20, fontweight='bold')

        # Recency分析
        r_counts = self.df['R_Stage'].value_counts()
        axes[0].pie(r_counts.values, labels=r_counts.index, autopct='%1.1f%%')
        axes[0].set_title('用户活跃度分布 (R)')

        # Frequency分析
        f_counts = self.df['F_Stage'].value_counts()
        axes[1].pie(f_counts.values, labels=f_counts.index, autopct='%1.1f%%')
        axes[1].set_title('购买频率分布 (F)')

        # Monetary分析
        m_counts = self.df['M_Stage'].value_counts()
        axes[2].pie(m_counts.values, labels=m_counts.index, autopct='%1.1f%%')
        axes[2].set_title('消费价值分布 (M)')

        plt.tight_layout()
        output_file = self.output_dir / "RFM分析结果.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def plot_cross_analysis(self):
        """绘制交叉分析图表"""
        # 检查必要的列
        required_cols = ['性别', '年龄层次', '收入等级']
        available_cols = [col for col in required_cols if col in self.df.columns]

        if len(available_cols) < 2:
            print("❌ 数据中缺少足够的交叉分析列")
            return

        # 创建交叉分析图表
        n_cols = len(available_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))

        if n_cols == 1:
            axes = [axes]

        fig.suptitle('用户特征交叉分析', fontsize=16, fontweight='bold')

        for i, col1 in enumerate(available_cols[:2]):  # 最多显示2个交叉分析
            col2 = '综合分群' if '综合分群' in self.df.columns else '收入等级'

            # 创建交叉表
            cross_table = pd.crosstab(self.df[col1], self.df[col2])

            # 绘制热力图
            sns.heatmap(cross_table, annot=True, fmt='d', cmap='YlOrRd', ax=axes[i])
            axes[i].set_title(f'{col1} vs {col2}')
            axes[i].set_xlabel(col2)
            axes[i].set_ylabel(col1)

        plt.tight_layout()
        output_file = self.output_dir / "交叉分析热力图.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def plot_income_consumption_scatter(self):
        """绘制收入-消费散点图"""
        if not all(col in self.df.columns for col in ['年收入', '年消费']):
            print("❌ 数据中缺少收入或消费列")
            return

        plt.figure(figsize=(12, 8))

        # 按用户分群着色
        if '综合分群' in self.df.columns:
            segments = self.df['综合分群'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))

            for segment, color in zip(segments, colors):
                segment_data = self.df[self.df['综合分群'] == segment]
                plt.scatter(segment_data['年收入'], segment_data['年消费'],
                           label=segment, alpha=0.7, color=color, s=50)

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(self.df['年收入'], self.df['年消费'], alpha=0.7, s=50)

        plt.xlabel('年收入 (元)')
        plt.ylabel('年消费 (元)')
        plt.title('收入-消费关系散点图', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 添加理想线 (消费=收入的10%)
        max_income = self.df['年收入'].max()
        plt.plot([0, max_income], [0, max_income * 0.1], 'r--', alpha=0.8, label='理想消费线(10%)')

        plt.tight_layout()
        output_file = self.output_dir / "收入消费散点图.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def generate_summary_dashboard(self):
        """生成综合仪表板"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('用户画像分析仪表板', fontsize=24, fontweight='bold')

        # 1. 用户分群分布 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        if '综合分群' in self.df.columns:
            segment_counts = self.df['综合分群'].value_counts()
            ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
            ax1.set_title('用户分群分布')

        # 2. 性别分布 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        if '性别' in self.df.columns:
            gender_counts = self.df['性别'].value_counts()
            ax2.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            ax2.set_title('性别分布')

        # 3. 年龄分布 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        if '年龄' in self.df.columns:
            ax3.hist(self.df['年龄'], bins=15, alpha=0.7, color='lightblue')
            ax3.set_title('年龄分布')
            ax3.set_xlabel('年龄')

        # 4. 收入分布 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        if '年收入' in self.df.columns:
            ax4.hist(self.df['年收入'] / 10000, bins=15, alpha=0.7, color='lightgreen')
            ax4.set_title('年收入分布')
            ax4.set_xlabel('年收入 (万元)')

        # 5. 消费分布 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        if '年消费' in self.df.columns:
            ax5.hist(self.df['年消费'], bins=15, alpha=0.7, color='salmon')
            ax5.set_title('年消费分布')
            ax5.set_xlabel('年消费 (元)')

        # 6. RFM分布 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'R_Stage' in self.df.columns:
            r_counts = self.df['R_Stage'].value_counts()
            ax6.bar(range(len(r_counts)), r_counts.values)
            ax6.set_xticks(range(len(r_counts)))
            ax6.set_xticklabels(r_counts.index, rotation=45)
            ax6.set_title('用户活跃度分布')

        # 7. 各分群平均消费 (左下)
        ax7 = fig.add_subplot(gs[2, :2])
        if '综合分群' in self.df.columns and '年消费' in self.df.columns:
            segment_consumption = self.df.groupby('综合分群')['年消费'].mean().sort_values()
            bars = ax7.barh(range(len(segment_consumption)), segment_consumption.values)
            ax7.set_yticks(range(len(segment_consumption)))
            ax7.set_yticklabels(segment_consumption.index)
            ax7.set_xlabel('平均年消费 (元)')
            ax7.set_title('各用户群体平均消费')

        # 8. 关键指标 (右下)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        # 计算关键指标
        metrics_text = f"""关键指标:

总用户数: {len(self.df):,}
平均年龄: {self.df['年龄'].mean():.1f}岁
平均年收入: {self.df['年收入'].mean()/10000:.1f}万元
平均年消费: {self.df['年消费'].mean():.1f}元
平均下单次数: {self.df['下单次数'].mean():.1f}次"""

        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')

        output_file = self.output_dir / "用户画像仪表板.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {output_file}")

    def run_visualization(self):
        """运行所有可视化"""
        print("🎨 开始生成用户画像可视化图表...")

        if not self.load_data():
            return False

        # 生成各种图表
        self.plot_segment_distribution()
        self.plot_consumption_by_segment()
        self.plot_income_by_segment()
        self.plot_demographic_distribution()
        self.plot_rfm_analysis()
        self.plot_cross_analysis()
        self.plot_income_consumption_scatter()
        self.generate_summary_dashboard()

        print("🎉 所有图表生成完成!")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='用户画像可视化工具')
    parser.add_argument('--data', required=True, help='分析结果数据文件路径')
    parser.add_argument('--output', default='visualization_output', help='图表输出目录')

    args = parser.parse_args()

    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        return 1

    # 创建可视化器并运行
    visualizer = UserProfilerVisualizer(args.data, args.output)

    success = visualizer.run_visualization()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())