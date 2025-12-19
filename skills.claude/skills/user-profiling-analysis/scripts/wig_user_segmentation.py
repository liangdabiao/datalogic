#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假发用户画像分析脚本
针对月收入、月消费数据进行用户分群和价值分析
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

class WigUserProfileAnalyzer:
    """假发用户画像分析器"""

    def __init__(self, data_path, output_dir="wig_analysis_output"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None

    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"SUCCESS: 数据加载成功: {self.df.shape}")
            return True
        except Exception as e:
            print(f"ERROR: 数据加载失败: {e}")
            return False

    def validate_data(self):
        """验证数据格式"""
        required_columns = ['用户编号', '年龄', '性别', '月收入', '月消费', '下单次数', '已注册月']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            print(f"ERROR: 缺失必要字段: {missing_columns}")
            return False

        print(f"SUCCESS: 数据验证通过")
        return True

    def preprocess_data(self):
        """数据预处理"""
        # 处理缺失值
        self.df = self.df.dropna()

        # 数据类型转换
        numeric_columns = ['年龄', '月收入', '月消费', '下单次数', '已注册月']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 添加衍生字段
        self.df['收入消费比'] = self.df['月消费'] / self.df['月收入']

        # 年龄分层
        def age_group(age):
            if age <= 27:
                return 'Z世代(18-27)'
            elif age <= 38:
                return '千禧一代(28-38)'
            else:
                return '成熟用户(39+)'

        self.df['年龄层次'] = self.df['年龄'].apply(age_group)

        # 收入分层 (月收入)
        def income_group(income):
            if income <= 3000:
                return '低收入'
            elif income <= 6000:
                return '中收入'
            else:
                return '高收入'

        self.df['收入等级'] = self.df['月收入'].apply(income_group)

        # 消费分层 (月消费)
        def consumption_group(consumption):
            if consumption <= 100:
                return '低消费'
            elif consumption <= 300:
                return '中消费'
            else:
                return '高消费'

        self.df['消费等级'] = self.df['月消费'].apply(consumption_group)

        print(f"SUCCESS: 数据预处理完成: {self.df.shape}")

    def wig_rfm_analysis(self, r_thresholds=(3, 6, 9), f_thresholds=(3, 5), m_thresholds=(100, 300)):
        """假发专用RFM分析"""

        # R分群 (注册时长)
        def r_stage(months):
            if months <= r_thresholds[0]:
                return '新用户'
            elif months <= r_thresholds[1]:
                return '活跃用户'
            elif months <= r_thresholds[2]:
                return '稳定用户'
            else:
                return '老用户'

        self.df['R_Stage'] = self.df['已注册月'].apply(r_stage)

        # F分群 (下单频率)
        def f_stage(orders):
            if orders <= f_thresholds[0]:
                return '低频'
            elif orders <= f_thresholds[1]:
                return '中频'
            else:
                return '高频'

        self.df['F_Stage'] = self.df['下单次数'].apply(f_stage)

        # M分群 (月消费金额)
        def m_stage(consumption):
            if consumption <= m_thresholds[0]:
                return '低价值'
            elif consumption <= m_thresholds[1]:
                return '中价值'
            else:
                return '高价值'

        self.df['M_Stage'] = self.df['月消费'].apply(m_stage)

        print("SUCCESS: 假发RFM分析完成")

    def wig_user_value_analysis(self):
        """假发用户价值分析"""

        def value_category(row):
            if row['收入等级'] == '高收入' and row['消费等级'] == '高消费':
                return '高价值用户'
            elif row['收入等级'] == '高收入' and row['消费等级'] == '低消费':
                return '潜力用户'
            elif row['收入等级'] == '中收入' and row['消费等级'] == '中消费':
                return '价值用户'
            elif row['收入消费比'] > 0.05:  # 消费占收入5%以上
                return '价格敏感'
            else:
                return '其他'

        self.df['用户价值'] = self.df.apply(value_category, axis=1)
        print("SUCCESS: 假发用户价值分析完成")

    def wig_comprehensive_segmentation(self):
        """假发综合用户分群"""

        def segment_user(row):
            # 假发用户特殊分群逻辑
            product = str(row.get('近期购买产品', '')).lower()

            # 年轻时尚假发用户
            if row['年龄'] <= 27 and '珠光眼影' in product:
                return '年轻时尚达人'

            # 高端假发用户
            elif row['月收入'] > 8000 and '假发' in product:
                return '高端假发追求者'

            # 护眼+假发用户
            elif '滴眼液' in product and row['年龄'] >= 28:
                return '品质生活追求者'

            # 学生党用户
            elif row['月收入'] <= 3000 and row['月消费'] <= 100:
                return '价格敏感学生党'

            # 忠实高价值用户
            elif row['已注册月'] >= 8 and row['月消费'] >= 200:
                return '忠实高价值用户'

            # 新兴潜力用户
            elif row['已注册月'] <= 3 and row['月收入'] >= 6000:
                return '新兴潜力用户'

            # 中年护眼群体
            elif row['年龄'] >= 28 and ('滴眼液' in product or '假发' in product):
                return '中年护理群体'

            else:
                return '其他用户群体'

        self.df['综合分群'] = self.df.apply(segment_user, axis=1)
        print("SUCCESS: 假发综合用户分群完成")

    def generate_basic_stats(self):
        """生成基础统计信息"""
        stats = {
            '总用户数': len(self.df),
            '平均年龄': self.df['年龄'].mean(),
            '平均月收入': self.df['月收入'].mean(),
            '平均月消费': self.df['月消费'].mean(),
            '平均下单次数': self.df['下单次数'].mean(),
            '平均注册月数': self.df['已注册月'].mean(),
            '性别分布': self.df['性别'].value_counts().to_dict(),
            '收入分布': self.df['收入等级'].value_counts().to_dict(),
            '消费分布': self.df['消费等级'].value_counts().to_dict(),
            '用户分群分布': self.df['综合分群'].value_counts().to_dict()
        }

        return stats

    def save_results(self):
        """保存分析结果"""
        # 保存分群结果
        output_file = self.output_dir / "假发用户分群结果.csv"
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"SUCCESS: 分群结果已保存: {output_file}")

        # 保存统计信息
        stats = self.generate_basic_stats()
        stats_file = self.output_dir / "假发用户统计信息.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== 假发用户画像分析基础统计 ===\n\n")
            f.write(f"总用户数: {stats['总用户数']:.2f}\n\n")
            f.write(f"平均年龄: {stats['平均年龄']:.2f}\n\n")
            f.write(f"平均月收入: {stats['平均月收入']:.2f}\n\n")
            f.write(f"平均月消费: {stats['平均月消费']:.2f}\n\n")
            f.write(f"平均下单次数: {stats['平均下单次数']:.2f}\n\n")
            f.write(f"平均注册月数: {stats['平均注册月数']:.2f}\n\n")

            f.write("性别分布:\n")
            for gender, count in stats['性别分布'].items():
                f.write(f"  {gender}: {count}\n")

            f.write("\n收入分布:\n")
            for level, count in stats['收入分布'].items():
                f.write(f"  {level}: {count}\n")

            f.write("\n消费分布:\n")
            for level, count in stats['消费分布'].items():
                f.write(f"  {level}: {count}\n")

            f.write("\n用户分群分布:\n")
            for segment, count in stats['用户分群分布'].items():
                f.write(f"  {segment}: {count}\n")

        print(f"SUCCESS: 统计信息已保存: {stats_file}")

    def run_full_analysis(self, r_thresholds=(3, 6, 9), f_thresholds=(3, 5), m_thresholds=(100, 300)):
        """运行完整分析流程"""
        print("开始假发用户画像分析...")

        if not self.load_data():
            return False

        if not self.validate_data():
            return False

        self.preprocess_data()
        self.wig_rfm_analysis(r_thresholds, f_thresholds, m_thresholds)
        self.wig_user_value_analysis()
        self.wig_comprehensive_segmentation()
        self.save_results()

        print("SUCCESS: 假发用户画像分析完成!")
        return True

def main():
    parser = argparse.ArgumentParser(description='假发用户画像分析工具')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--output', default='wig_analysis_output', help='输出目录')
    parser.add_argument('--r-thresholds', nargs=3, type=int, default=[3, 6, 9],
                       help='R分群阈值 (默认: 3 6 9)')
    parser.add_argument('--f-thresholds', nargs=2, type=int, default=[3, 5],
                       help='F分群阈值 (默认: 3 5)')
    parser.add_argument('--m-thresholds', nargs=2, type=int, default=[100, 300],
                       help='M分群阈值 (默认: 100 300)')

    args = parser.parse_args()

    analyzer = WigUserProfileAnalyzer(args.data, args.output)
    success = analyzer.run_full_analysis(
        tuple(args.r_thresholds),
        tuple(args.f_thresholds),
        tuple(args.m_thresholds)
    )

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())