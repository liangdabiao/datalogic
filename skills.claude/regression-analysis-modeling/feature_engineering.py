#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Feature Engineering Tools
Comprehensive feature engineering for regression analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Advanced feature engineering toolkit for regression analysis"""

    def __init__(self):
        """Initialize feature engineering toolkit"""
        self.scalers = {}
        self.encoders = {}
        self.feature_transformers = {}
        self.feature_selection_results = {}

    def extract_temporal_features(self, df, date_columns, reference_date=None):
        """
        Extract comprehensive temporal features from date columns

        Args:
            df (pd.DataFrame): Input dataframe
            date_columns (list): List of date column names
            reference_date (datetime): Reference date for calculations

        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        print("\n=== 时间特征工程 ===")

        df_temporal = df.copy()

        if reference_date is None:
            reference_date = datetime.now()

        for col in date_columns:
            if col in df_temporal.columns:
                # Ensure datetime format
                df_temporal[col] = pd.to_datetime(df_temporal[col], errors='coerce')

                # Basic temporal features
                df_temporal[f'{col}_year'] = df_temporal[col].dt.year
                df_temporal[f'{col}_month'] = df_temporal[col].dt.month
                df_temporal[f'{col}_day'] = df_temporal[col].dt.day
                df_temporal[f'{col}_dayofweek'] = df_temporal[col].dt.dayofweek
                df_temporal[f'{col}_dayofyear'] = df_temporal[col].dt.dayofyear
                df_temporal[f'{col}_quarter'] = df_temporal[col].dt.quarter
                df_temporal[f'{col}_week'] = df_temporal[col].dt.isocalendar().week

                # Cyclical features for better seasonality capture
                df_temporal[f'{col}_month_sin'] = np.sin(2 * np.pi * df_temporal[f'{col}_month'] / 12)
                df_temporal[f'{col}_month_cos'] = np.cos(2 * np.pi * df_temporal[f'{col}_month'] / 12)
                df_temporal[f'{col}_day_sin'] = np.sin(2 * np.pi * df_temporal[f'{col}_day'] / 31)
                df_temporal[f'{col}_day_cos'] = np.cos(2 * np.pi * df_temporal[f'{col}_day'] / 31)

                # Time since reference date
                df_temporal[f'{col}_days_since_ref'] = (reference_date - df_temporal[col]).dt.days
                df_temporal[f'{col}_weeks_since_ref'] = df_temporal[f'{col}_days_since_ref'] / 7
                df_temporal[f'{col}_months_since_ref'] = df_temporal[f'{col}_days_since_ref'] / 30.44

                # Weekend and holiday indicators
                df_temporal[f'{col}_is_weekend'] = (df_temporal[f'{col}_dayofweek'] >= 5).astype(int)

                print(f"- 为 {col} 创建 {14} 个时间特征")

        return df_temporal

    def create_rfm_analysis(self, df, user_id_col, date_col, amount_col,
                           analysis_period_days=365, segment_customers=True):
        """
        Advanced RFM (Recency, Frequency, Monetary) analysis with customer segmentation

        Args:
            df (pd.DataFrame): Transaction data
            user_id_col (str): User identifier column
            date_col (str): Transaction date column
            amount_col (str): Transaction amount column
            analysis_period_days (int): Analysis period in days
            segment_customers (bool): Whether to create customer segments

        Returns:
            pd.DataFrame: RFM analysis with customer segments
        """
        print(f"\n=== 高级RFM分析 (分析期间: {analysis_period_days}天) ===")

        # Ensure proper data types
        df_rfm = df.copy()
        df_rfm[date_col] = pd.to_datetime(df_rfm[date_col])
        df_rfm[amount_col] = pd.to_numeric(df_rfm[amount_col], errors='coerce')

        # Calculate analysis period
        max_date = df_rfm[date_col].max()
        analysis_start = max_date - pd.Timedelta(days=analysis_period_days)

        # Filter data
        df_analysis = df_rfm[df_rfm[date_col] >= analysis_start]

        # Calculate basic RFM metrics
        rfm_data = []

        for user_id in df_rfm[user_id_col].unique():
            user_data = df_analysis[df_analysis[user_id_col] == user_id]

            if len(user_data) == 0:
                continue

            # Basic RFM
            recency = (max_date - user_data[date_col].max()).days
            frequency = len(user_data)
            monetary = user_data[amount_col].sum()

            # Advanced metrics
            avg_order_value = monetary / frequency if frequency > 0 else 0
            first_purchase = user_data[date_col].min()
            customer_lifetime = (max_date - first_purchase).days
            purchase_frequency = frequency / (customer_lifetime / 30) if customer_lifetime > 0 else 0

            # Purchase pattern metrics
            purchase_intervals = user_data[date_col].sort_values().diff().dt.days.dropna()
            avg_purchase_interval = purchase_intervals.mean() if len(purchase_intervals) > 0 else 0
            std_purchase_interval = purchase_intervals.std() if len(purchase_intervals) > 1 else 0

            # Monetary pattern metrics
            order_values = user_data[amount_col]
            avg_order_size = order_values.mean()
            std_order_size = order_values.std() if len(order_values) > 1 else 0

            # Recent activity metrics (last 30/60/90 days)
            recent_30 = user_data[user_data[date_col] >= (max_date - pd.Timedelta(days=30))]
            recent_60 = user_data[user_data[date_col] >= (max_date - pd.Timedelta(days=60))]
            recent_90 = user_data[user_data[date_col] >= (max_date - pd.Timedelta(days=90))]

            recency_30 = len(recent_30)
            recency_60 = len(recent_60)
            recency_90 = len(recent_90)

            monetary_30 = recent_30[amount_col].sum()
            monetary_60 = recent_60[amount_col].sum()
            monetary_90 = recent_90[amount_col].sum()

            # Customer stability metrics
            active_months = user_data[date_col].dt.to_period('M').nunique()
            purchase_consistency = frequency / active_months if active_months > 0 else 0

            rfm_data.append({
                user_id_col: user_id,
                'R_最近购买天数': recency,
                'F_总购买频次': frequency,
                'M_总消费金额': monetary,
                'AOV_平均客单价': avg_order_value,
                'CLV_客户生命周期': customer_lifetime,
                'PF_购买频率': purchase_frequency,
                'API_平均购买间隔': avg_purchase_interval,
                'SPI_购买间隔稳定性': std_purchase_interval,
                'AOS_平均订单规模': avg_order_size,
                'SOS_订单规模稳定性': std_order_size,
                'R30_近30天购买次数': recency_30,
                'R60_近60天购买次数': recency_60,
                'R90_近90天购买次数': recency_90,
                'M30_近30天消费金额': monetary_30,
                'M60_近60天消费金额': monetary_60,
                'M90_近90天消费金额': monetary_90,
                'AM_活跃月份数': active_months,
                'PC_购买一致性': purchase_consistency
            })

        rfm_df = pd.DataFrame(rfm_data)

        if segment_customers and len(rfm_df) > 0:
            rfm_df = self._create_customer_segments(rfm_df)

        print(f"为 {len(rfm_df)} 个用户生成 {len(rfm_df.columns)-1} 个RFM特征")
        return rfm_df

    def _create_customer_segments(self, rfm_df):
        """Create customer segments based on RFM scores"""
        print("- 创建客户分群")

        # Calculate RFM scores (1-5 scale)
        for metric in ['R_最近购买天数', 'F_总购买频次', 'M_总消费金额']:
            score_col = f'{metric}_score'

            if metric == 'R_最近购买天数':
                # Lower recency is better
                rfm_df[score_col] = pd.qcut(rfm_df[metric], q=5, labels=False, duplicates='drop') + 1
                rfm_df[score_col] = 6 - rfm_df[score_col]  # Reverse scoring
            else:
                # Higher frequency and monetary are better
                rfm_df[score_col] = pd.qcut(rfm_df[metric], q=5, labels=False, duplicates='drop') + 1

        # Calculate total RFM score
        rfm_df['RFM_total_score'] = (
            rfm_df['R_最近购买天数_score'] +
            rfm_df['F_总购买频次_score'] +
            rfm_df['M_总消费金额_score']
        )

        # Create segments based on RFM scores
        def assign_segment(score):
            if score >= 13:
                return 'Champions_冠军客户'
            elif score >= 11:
                return 'Loyal_Customers_忠诚客户'
            elif score >= 9:
                return 'Potential_Loyalist_潜力客户'
            elif score >= 7:
                return 'New_Customers_新客户'
            elif score >= 5:
                return 'At_Risk_Clients_风险客户'
            else:
                return 'Lost_Customers_流失客户'

        rfm_df['Customer_Segment'] = rfm_df['RFM_total_score'].apply(assign_segment)

        # Create value tiers
        rfm_df['Value_Tier'] = pd.cut(
            rfm_df['RFM_total_score'],
            bins=[0, 6, 10, 15],
            labels=['Low_Value_低价值', 'Medium_Value_中等价值', 'High_Value_高价值']
        )

        return rfm_df

    def create_polynomial_features(self, X, degree=2, interaction_only=False):
        """
        Create polynomial features for non-linear relationships

        Args:
            X (pd.DataFrame): Input features
            degree (int): Polynomial degree
            interaction_only (bool): Create only interaction terms

        Returns:
            pd.DataFrame: DataFrame with polynomial features
        """
        print(f"\n=== 多项式特征工程 (degree={degree}) ===")

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            print("- 没有数值型特征可用于多项式变换")
            return X.copy()

        # Select features for polynomial transformation (avoid too many features)
        if len(numerical_cols) > 8:
            # Select features with highest variance
            variances = X[numerical_cols].var()
            selected_features = variances.nlargest(8).index
            print(f"- 从 {len(numerical_cols)} 个数值特征中选择方差最高的 8 个")
        else:
            selected_features = numerical_cols

        X_numerical = X[selected_features]

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        X_poly = poly.fit_transform(X_numerical)

        # Get feature names
        poly_feature_names = poly.get_feature_names_out(selected_features)

        # Create DataFrame
        poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)

        # Combine with original features
        X_combined = pd.concat([X.drop(columns=selected_features), poly_df], axis=1)

        print(f"- 创建了 {len(poly_feature_names)} 个多项式特征")
        print(f"- 总特征数: {X_combined.shape[1]}")

        return X_combined

    def create_aggregation_features(self, df, group_columns, agg_columns,
                                  agg_functions=['mean', 'std', 'min', 'max', 'count']):
        """
        Create aggregation features for grouped data

        Args:
            df (pd.DataFrame): Input dataframe
            group_columns (list): Columns to group by
            agg_columns (list): Columns to aggregate
            agg_functions (list): Aggregation functions

        Returns:
            pd.DataFrame: DataFrame with aggregation features
        """
        print(f"\n=== 聚合特征工程 ===")

        if not all(col in df.columns for col in group_columns):
            print("- 分组列不存在，跳过聚合特征")
            return df.copy()

        agg_df = df.copy()

        # Create aggregation features
        for group_col in group_columns:
            if group_col not in df.columns:
                continue

            for agg_col in agg_columns:
                if agg_col not in df.columns:
                    continue

                for func in agg_functions:
                    if func == 'count':
                        # Count is special case
                        feature_name = f"{group_col}_{agg_col}_count"
                        agg_df[feature_name] = df.groupby(group_col)[agg_col].transform('count')
                    else:
                        feature_name = f"{group_col}_{agg_col}_{func}"
                        agg_df[feature_name] = df.groupby(group_col)[agg_col].transform(func)

        print(f"- 为分组列 {group_columns} 创建聚合特征")
        print(f"- 新增特征数: {agg_df.shape[1] - df.shape[1]}")

        return agg_df

    def create_ratio_features(self, X, ratio_pairs):
        """
        Create ratio features from pairs of columns

        Args:
            X (pd.DataFrame): Input features
            ratio_pairs (list): List of tuples (numerator, denominator, new_name)

        Returns:
            pd.DataFrame: DataFrame with ratio features
        """
        print(f"\n=== 比例特征工程 ===")

        X_ratios = X.copy()
        created_features = 0

        for numerator, denominator, new_name in ratio_pairs:
            if numerator in X.columns and denominator in X.columns:
                # Handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    X_ratios[new_name] = np.where(
                        X[denominator] != 0,
                        X[numerator] / X[denominator],
                        0  # or np.nan, depending on preference
                    )
                created_features += 1
                print(f"- 创建比例特征: {new_name} = {numerator} / {denominator}")
            else:
                print(f"- 跳过 {new_name}: 缺少必要的列")

        print(f"- 总共创建了 {created_features} 个比例特征")
        return X_ratios

    def create_binning_features(self, X, binning_config):
        """
        Create binned/discretized features

        Args:
            X (pd.DataFrame): Input features
            binning_config (dict): Configuration for binning

        Returns:
            pd.DataFrame: DataFrame with binned features
        """
        print(f"\n=== 分箱特征工程 ===")

        X_binned = X.copy()

        for col, config in binning_config.items():
            if col not in X.columns:
                print(f"- 跳过 {col}: 列不存在")
                continue

            if isinstance(config, dict):
                bin_type = config.get('type', 'quantile')
                bins = config.get('bins', 5)
                labels = config.get('labels', None)
                suffix = config.get('suffix', '_binned')
            else:
                # Simple configuration
                bin_type = 'quantile'
                bins = config
                labels = None
                suffix = '_binned'

            try:
                if bin_type == 'quantile':
                    X_binned[f"{col}{suffix}"] = pd.qcut(X[col], q=bins, labels=labels, duplicates='drop')
                elif bin_type == 'uniform':
                    X_binned[f"{col}{suffix}"] = pd.cut(X[col], bins=bins, labels=labels)
                else:
                    print(f"- 不支持的分箱类型: {bin_type}")
                    continue

                print(f"- 为 {col} 创建分箱特征 (类型: {bin_type}, 分箱数: {bins})")

            except Exception as e:
                print(f"- 为 {col} 创建分箱特征失败: {str(e)}")

        return X_binned

    def select_features(self, X, y, method='f_regression', k=10):
        """
        Feature selection using statistical methods

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            method (str): Selection method
            k (int): Number of features to select

        Returns:
            tuple: (selected_features, selection_scores)
        """
        print(f"\n=== 特征选择 (method={method}, k={k}) ===")

        # Ensure X is numeric
        X_numeric = X.select_dtypes(include=[np.number])

        if X_numeric.shape[1] == 0:
            print("- 没有数值型特征可供选择")
            return X, {}

        # Feature selection
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, X_numeric.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X_numeric.shape[1]))
        else:
            raise ValueError(f"Unsupported selection method: {method}")

        # Fit selector
        X_selected = selector.fit_transform(X_numeric, y)
        selected_features = X_numeric.columns[selector.get_support()]
        selection_scores = selector.scores_[selector.get_support()]

        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': selected_features,
            'score': selection_scores
        }).sort_values('score', ascending=False)

        # Create selected features DataFrame
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        # Add non-numeric columns back
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            X_selected_df[col] = X[col]

        print(f"- 从 {X_numeric.shape[1]} 个特征中选择了 {len(selected_features)} 个")
        print("- Top 10 特征:")
        for _, row in results_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['score']:.4f}")

        self.feature_selection_results[method] = results_df

        return X_selected_df, results_df

    def scale_features(self, X, method='standard', columns=None):
        """
        Scale numerical features

        Args:
            X (pd.DataFrame): Input features
            method (str): Scaling method
            columns (list): Columns to scale (all numeric if None)

        Returns:
            pd.DataFrame: Scaled features
        """
        print(f"\n=== 特征缩放 (method={method}) ===")

        X_scaled = X.copy()

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        # Fit and transform
        X_scaled[columns] = scaler.fit_transform(X[columns])

        # Store scaler
        self.scalers[method] = scaler

        print(f"- 使用 {method} 方法缩放 {len(columns)} 个数值特征")

        return X_scaled

def main():
    """Example usage"""
    fe = FeatureEngineering()

    # Create sample data
    sample_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'date': pd.date_range('2023-01-01', periods=5),
        'amount': [100, 150, 200, 120, 80],
        'category': ['A', 'B', 'A', 'C', 'B']
    })

    # Test RFM analysis
    rfm_result = fe.create_rfm_analysis(
        sample_data,
        user_id_col='user_id',
        date_col='date',
        amount_col='amount'
    )
    print("RFM分析结果:")
    print(rfm_result)

if __name__ == "__main__":
    main()