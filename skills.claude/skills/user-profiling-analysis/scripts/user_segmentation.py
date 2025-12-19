#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户画像分析主脚本
基于消费数据进行用户分群和价值分析
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# 可视化相关库，如果不存在则跳过
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_VISUALIZATION = False
    print("WARNING: Visualization libraries not installed, skipping chart generation")

# 设置中文字体支持 (如果可视化库可用)
if HAS_VISUALIZATION:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

class UserProfileAnalyzer:
    """用户画像分析器"""

    def __init__(self, data_path, output_dir="analysis_output"):
        """
        初始化分析器

        Args:
            data_path: 数据文件路径
            output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None

    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"SUCCESS: Data loaded successfully: {self.df.shape}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            return False

    def validate_data(self):
        """验证数据格式"""
        required_columns = ['用户编号', '年龄', '性别', '年收入', '年消费', '下单次数', '已注册月']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            return False

        print(f"SUCCESS: Data validation passed")
        return True

    def preprocess_data(self):
        """数据预处理"""
        # 处理缺失值
        self.df = self.df.dropna()

        # 数据类型转换
        numeric_columns = ['年龄', '年收入', '年消费', '下单次数', '已注册月']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 添加衍生字段
        self.df['收入消费比'] = self.df['年消费'] / (self.df['年收入'] + 1e-6)

        # 年龄分层
        def age_group(age):
            if age <= 27:
                return 'Z世代(18-27)'
            elif age <= 38:
                return '千禧一代(28-38)'
            else:
                return '成熟用户(39+)'

        self.df['年龄层次'] = self.df['年龄'].apply(age_group)

        # 收入分层
        def income_group(income):
            if income <= 30000:
                return '低收入'
            elif income <= 60000:
                return '中收入'
            else:
                return '高收入'

        self.df['收入等级'] = self.df['年收入'].apply(income_group)

        # 消费分层
        def consumption_group(cons):
            if cons <= 100:
                return '低消费'
            elif cons <= 200:
                return '中消费'
            else:
                return '高消费'

        self.df['消费等级'] = self.df['年消费'].apply(consumption_group)

        print(f"SUCCESS: Data preprocessing completed: {self.df.shape}")

    def rfm_analysis(self, r_thresholds=(3, 6, 9), f_thresholds=(3, 5), m_thresholds=(100, 200)):
        """RFM分析"""
        # R分群 (注册时长)
        def r_score(x):
            if x <= r_thresholds[0]:
                return '新用户'
            elif x <= r_thresholds[1]:
                return '活跃'
            elif x <= r_thresholds[2]:
                return '稳定'
            else:
                return '老用户'

        # F分群 (下单频率)
        def f_score(x):
            if x <= f_thresholds[0]:
                return '低频'
            elif x <= f_thresholds[1]:
                return '中频'
            else:
                return '高频'

        # M分群 (消费金额)
        def m_score(x):
            if x <= m_thresholds[0]:
                return '低价值'
            elif x <= m_thresholds[1]:
                return '中价值'
            else:
                return '高价值'

        self.df['R_Stage'] = self.df['已注册月'].apply(r_score)
        self.df['F_Stage'] = self.df['下单次数'].apply(f_score)
        self.df['M_Stage'] = self.df['年消费'].apply(m_score)

        print("SUCCESS: RFM analysis completed")

    def value_analysis(self):
        """用户价值分析"""
        def value_score(row):
            if row['收入等级'] == '高收入' and row['消费等级'] == '高消费':
                return '高价值用户'
            elif row['收入等级'] == '高收入' and row['消费等级'] == '低消费':
                return '潜力用户'
            elif row['收入等级'] == '中收入' and row['消费等级'] == '中消费':
                return '价值用户'
            elif row['收入等级'] == '低收入' and row['收入消费比'] > self.df['收入消费比'].median():
                return '价格敏感'
            else:
                return '其他'

        self.df['用户价值'] = self.df.apply(value_score, axis=1)
        print("SUCCESS: User value analysis completed")

    def comprehensive_segmentation(self):
        """综合用户分群"""
        def segment_user(row):
            # 检查是否存在产品相关字段
            product_col = None
            for col in ['近期购买产品', '最近购买产品', '产品偏好']:
                if col in self.df.columns:
                    product_col = col
                    break

            if product_col is None:
                # 如果没有产品信息，基于基础特征分群
                if row['年龄层次'] == 'Z世代(18-27)' and row['收入等级'] == '低收入':
                    return '年轻价格敏感群体'
                elif row['用户价值'] == '高价值用户':
                    return '忠诚高价值用户'
                elif row['R_Stage'] == '新用户' and row['收入等级'] == '高收入':
                    return '新兴潜力用户'
                else:
                    return '其他用户群体'
            else:
                product = row[product_col]

                # 年轻护眼刚需族
                if (row['年龄层次'] == 'Z世代(18-27)' and
                    any(keyword in str(product) for keyword in ['眼镜', '护眼', '视力'])):
                    return '年轻护眼刚需族'

                # 高收入美妆爱好者
                elif (row['收入等级'] == '高收入' and
                      any(keyword in str(product) for keyword in ['眼影', '美妆', '化妆品'])):
                    return '高收入美妆爱好者'

                # 中年视力关爱群体
                elif (row['年龄层次'] in ['千禧一代(28-38)', '成熟用户(39+)'] and
                      any(keyword in str(product) for keyword in ['眼镜', '护眼', '滴眼液'])):
                    return '中年视力关爱群体'

                # 价格敏感学生党
                elif (row['年龄层次'] == 'Z世代(18-27)' and
                      row['收入等级'] == '低收入' and
                      row['消费等级'] == '低消费'):
                    return '价格敏感学生党'

                # 忠诚高价值用户
                elif row['R_Stage'] in ['活跃', '稳定'] and row['用户价值'] == '高价值用户':
                    return '忠诚高价值用户'

                # 新兴潜力用户
                elif row['R_Stage'] == '新用户' and row['收入等级'] == '高收入':
                    return '新兴潜力用户'

                # 流失风险用户
                elif row['R_Stage'] == '老用户' and row['F_Stage'] == '低频':
                    return '流失风险用户'

                else:
                    return '其他用户群体'

        self.df['综合分群'] = self.df.apply(segment_user, axis=1)
        print("SUCCESS: Comprehensive user segmentation completed")

    def generate_basic_stats(self):
        """生成基础统计信息"""
        stats = {
            '总用户数': len(self.df),
            '平均年龄': self.df['年龄'].mean(),
            '平均年收入': self.df['年收入'].mean(),
            '平均年消费': self.df['年消费'].mean(),
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
        output_file = self.output_dir / "用户分群结果.csv"
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"SUCCESS: Segmentation results saved: {output_file}")

        # 保存统计信息
        stats = self.generate_basic_stats()
        stats_file = self.output_dir / "基础统计信息.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== User Profiling Analysis Basic Statistics ===\n\n")
            for key, value in stats.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value:.2f}\n")
                f.write("\n")

        print(f"SUCCESS: Statistics saved: {stats_file}")

    def run_full_analysis(self, r_thresholds=(3, 6, 9), f_thresholds=(3, 5), m_thresholds=(100, 200)):
        """运行完整分析流程"""
        print("Starting user profiling analysis...")

        if not self.load_data():
            return False

        if not self.validate_data():
            return False

        self.preprocess_data()
        self.rfm_analysis(r_thresholds, f_thresholds, m_thresholds)
        self.value_analysis()
        self.comprehensive_segmentation()
        self.save_results()

        print("SUCCESS: User profiling analysis completed!")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='用户画像分析工具')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--output', default='analysis_output', help='输出目录')
    parser.add_argument('--r-thresholds', nargs=3, type=int, default=[3, 6, 9],
                       help='RFM分析中R分组的阈值 (新用户/活跃/稳定的最大月数)')
    parser.add_argument('--f-thresholds', nargs=2, type=int, default=[3, 5],
                       help='RFM分析中F分组的阈值 (低频/中频的最大次数)')
    parser.add_argument('--m-thresholds', nargs=2, type=int, default=[100, 200],
                       help='RFM分析中M分组的阈值 (低价值/中价值的最大金额)')

    args = parser.parse_args()

    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        return 1

    # 创建分析器并运行分析
    analyzer = UserProfileAnalyzer(args.data, args.output)

    success = analyzer.run_full_analysis(
        r_thresholds=args.r_thresholds,
        f_thresholds=args.f_thresholds,
        m_thresholds=args.m_thresholds
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())