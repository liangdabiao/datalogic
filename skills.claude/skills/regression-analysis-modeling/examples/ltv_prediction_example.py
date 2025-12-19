#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer LTV Prediction Example
Complete example of customer lifetime value prediction using RFM analysis
"""

import pandas as pd
import numpy as np
from core_regression import RegressionAnalyzer
from feature_engineering import FeatureEngineering
from model_evaluation import ModelEvaluator
from prediction_visualizer import PredictionVisualizer

def create_sample_ltv_data():
    """Create sample customer transaction data for LTV prediction"""
    print("创建示例LTV数据...")

    # Create customer base
    np.random.seed(42)
    n_customers = 500
    customer_ids = [f"CUST{i:04d}" for i in range(1, n_customers + 1)]

    # Create customer profiles
    customers = []
    for customer_id in customer_ids:
        # Customer characteristics
        registration_month = np.random.randint(1, 13)
        registration_year = np.random.choice([2022, 2023], p=[0.3, 0.7])
        customer_type = np.random.choice(['新客户', '老客户'], p=[0.6, 0.4])
        city = np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都'],
                              p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15])

        customers.append({
            '客户ID': customer_id,
            '注册年份': registration_year,
            '注册月份': registration_month,
            '客户类型': customer_type,
            '城市': city
        })

    customer_df = pd.DataFrame(customers)

    # Generate transaction data
    transactions = []
    order_id = 1

    for _, customer in customer_df.iterrows():
        # Determine number of transactions based on customer type
        if customer['客户类型'] == '老客户':
            n_transactions = np.random.poisson(8)
        else:
            n_transactions = np.random.poisson(3)

        for _ in range(n_transactions):
            # Random transaction date within the last year
            days_ago = np.random.randint(1, 365)
            transaction_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=365 - days_ago)

            # Transaction amount varies by city and customer type
            city_multiplier = {'北京': 1.2, '上海': 1.3, '广州': 1.0, '深圳': 1.1, '杭州': 0.9, '成都': 0.8}
            type_multiplier = {'老客户': 1.2, '新客户': 0.8}

            base_amount = np.random.exponential(100) * city_multiplier[customer['城市']] * type_multiplier[customer['客户类型']]
            quantity = np.random.randint(1, 5)
            unit_price = base_amount / quantity

            transactions.append({
                '订单号': f"ORD{order_id:06d}",
                '用户码': customer['客户ID'],
                '消费日期': transaction_date.strftime('%Y-%m-%d'),
                '产品说明': np.random.choice(['电子产品', '服装', '食品', '家居用品', '图书']),
                '数量': quantity,
                '单价': round(unit_price, 2),
                '城市': customer['城市']
            })

            order_id += 1

    transaction_df = pd.DataFrame(transactions)

    # Calculate total amount for each transaction
    transaction_df['总价'] = transaction_df['数量'] * transaction_df['单价']

    # Calculate annual LTV for each customer
    ltv_data = []
    for customer_id in customer_ids:
        customer_transactions = transaction_df[transaction_df['用户码'] == customer_id]

        # Calculate RFM metrics
        recency = (pd.Timestamp('2024-12-31') - pd.to_datetime(customer_transactions['消费日期']).max()).days
        frequency = len(customer_transactions)
        monetary = customer_transactions['总价'].sum()

        # Calculate annual LTV (monetary is already for the year)
        annual_ltv = monetary

        # Add some noise and business logic
        if frequency > 10:
            annual_ltv *= 1.2  # High frequency customers get more value
        if recency < 30:
            annual_ltv *= 1.1  # Recent customers get slight boost

        ltv_data.append({
            '用户码': customer_id,
            '年度LTV': round(annual_ltv, 2)
        })

    ltv_df = pd.DataFrame(ltv_data)

    # Save sample data
    transaction_df.to_csv('sample_transactions.csv', index=False, encoding='utf-8-sig')
    ltv_df.to_csv('sample_ltv_targets.csv', index=False, encoding='utf-8-sig')

    print(f"生成了 {len(transaction_df)} 条交易记录，覆盖 {len(ltv_df)} 个客户")
    return transaction_df, ltv_df

def run_ltv_prediction_example():
    """Run complete LTV prediction example"""
    print("🚀 开始客户LTV预测示例")
    print("=" * 50)

    # 1. Create sample data
    transaction_df, ltv_df = create_sample_ltv_data()

    # 2. Initialize analyzers
    analyzer = RegressionAnalyzer()
    fe = FeatureEngineering()
    evaluator = ModelEvaluator()
    visualizer = PredictionVisualizer()

    # 3. Create RFM features
    print("\n=== RFM特征工程 ===")
    rfm_features = fe.create_rfm_analysis(
        transaction_df,
        user_id_col='用户码',
        date_col='消费日期',
        amount_col='总价',
        analysis_period_days=365
    )

    # 4. Merge RFM features with LTV targets
    print("\n=== 数据合并 ===")
    analysis_data = pd.merge(rfm_features, ltv_df, on='用户码')
    print(f"合并后数据形状: {analysis_data.shape}")

    # 5. Prepare features and target
    feature_cols = [col for col in analysis_data.columns
                   if col not in ['用户码', '年度LTV', 'Customer_Segment', 'Value_Tier']]
    X = analysis_data[feature_cols]
    y = analysis_data['年度LTV']

    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")

    # 6. Run regression analysis
    print("\n=== 回归模型训练 ===")
    analysis_results = analyzer.run_complete_analysis(
        analysis_data,
        '年度LTV',
        create_interactions=False  # RFM features are already comprehensive
    )

    # 7. Detailed model evaluation
    print("\n=== 模型详细评估 ===")

    # Residual analysis for best model
    best_model_name = analyzer.best_model_name
    best_results = analysis_results['results'][best_model_name]

    residual_analysis = evaluator.perform_residual_analysis(
        best_results['y_test'],
        best_results['predictions'],
        best_model_name
    )

    # Learning curve analysis
    learning_analysis = evaluator.analyze_learning_curves(
        analyzer.best_model,
        analysis_results['X_final'],
        analysis_results['y_final']
    )

    # 8. Create visualizations
    print("\n=== 生成可视化分析 ===")

    # Comprehensive dashboard
    visualizer.create_comprehensive_dashboard(
        analysis_results['results'],
        analysis_results['feature_importance'],
        save_path='ltv_analysis_dashboard.png'
    )

    # Individual analysis plots
    visualizer.create_individual_analysis_plots(
        analysis_results['results'],
        output_dir='ltv_analysis_plots'
    )

    # 9. Generate comprehensive report
    print("\n=== 生成分析报告 ===")
    evaluation_report = evaluator.generate_evaluation_report(
        analysis_results['results'],
        save_to_file=True
    )

    # 10. Business insights
    print("\n=== 业务洞察 ===")
    feature_importance = analysis_results['feature_importance']

    print("Top 5 影响LTV的关键因素:")
    for idx, row in feature_importance.head(5).iterrows():
        feature_name = row['feature']
        # Translate feature names for business interpretation
        feature_translation = {
            'M_总消费金额': '总消费金额',
            'F_总购买频次': '购买频次',
            'AOV_平均客单价': '平均客单价',
            'PF_购买频率': '购买频率',
            'CLV_客户生命周期': '客户活跃天数'
        }

        business_name = feature_translation.get(feature_name, feature_name)
        print(f"  {idx + 1}. {business_name}: {row['importance']:.4f}")

    # 11. Summary statistics
    print(f"\n=== 模型性能总结 ===")
    best_metrics = analysis_results['results'][best_model_name]['metrics']
    print(f"最佳模型: {best_model_name}")
    print(f"R² 分数: {best_metrics['test_r2']:.4f}")
    print(f"平均绝对误差: {best_metrics['test_mae']:.2f}")
    print(f"均方根误差: {best_metrics['test_rmse']:.2f}")

    # Calculate business metrics
    mae_percentage = (best_metrics['test_mae'] / y.mean()) * 100
    print(f"平均预测误差百分比: {mae_percentage:.2f}%")

    # 12. Recommendations
    print(f"\n=== 营销建议 ===")
    print("基于RFM分析的客户分层策略:")

    # Create customer segments based on RFM
    high_value_threshold = y.quantile(0.8)
    medium_value_threshold = y.quantile(0.5)

    high_value_customers = analysis_data[analysis_data['年度LTV'] >= high_value_threshold]
    medium_value_customers = analysis_data[(analysis_data['年度LTV'] >= medium_value_threshold) &
                                          (analysis_data['年度LTV'] < high_value_threshold)]
    low_value_customers = analysis_data[analysis_data['年度LTV'] < medium_value_threshold]

    print(f"- 高价值客户: {len(high_value_customers)} 人 ({len(high_value_customers)/len(analysis_data)*100:.1f}%)")
    print(f"- 中等价值客户: {len(medium_value_customers)} 人 ({len(medium_value_customers)/len(analysis_data)*100:.1f}%)")
    print(f"- 低价值客户: {len(low_value_customers)} 人 ({len(low_value_customers)/len(analysis_data)*100:.1f}%)")

    print("\n营销策略建议:")
    print("1. 高价值客户: VIP服务、个性化推荐、专属优惠")
    print("2. 中等价值客户: 交叉销售、升级促销、会员激励")
    print("3. 低价值客户: 重新激活、优惠刺激、价值教育")

    print(f"\n✅ LTV预测分析完成！")
    print(f"生成的文件:")
    print(f"- sample_transactions.csv: 示例交易数据")
    print(f"- sample_ltv_targets.csv: LTV目标数据")
    print(f"- ltv_analysis_dashboard.png: 综合分析仪表板")
    print(f"- ltv_analysis_plots/: 详细分析图表")
    print(f"- model_evaluation_report.md: 评估报告")

if __name__ == "__main__":
    run_ltv_prediction_example()