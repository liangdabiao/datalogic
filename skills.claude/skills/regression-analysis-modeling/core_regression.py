#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Regression Analysis Engine
Comprehensive regression modeling with multiple algorithms and automated feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class RegressionAnalyzer:
    """Comprehensive regression analysis engine with multiple algorithms"""

    def __init__(self, random_state=42, chinese_font='SimHei'):
        """Initialize the regression analyzer"""
        self.random_state = random_state
        self.chinese_font = chinese_font
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.target_name = None
        self.best_model = None
        self.best_model_name = None

    def load_and_validate_data(self, file_path, target_column, **kwargs):
        """
        Load and validate data for regression analysis

        Args:
            file_path (str): Path to the CSV file
            target_column (str): Name of the target variable column
            **kwargs: Additional parameters for pd.read_csv()

        Returns:
            tuple: (X, y) features and target
        """
        print("=== 数据加载与验证 ===")

        # Load data with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8', **kwargs)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk', **kwargs)
            except:
                df = pd.read_csv(file_path, encoding='latin-1', **kwargs)

        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # Check if target column exists
        if target_column not in df.columns:
            # Try to find similar column names
            possible_targets = [col for col in df.columns
                              if target_column.lower() in col.lower() or
                                 col.lower() in target_column.lower()]
            if possible_targets:
                target_column = possible_targets[0]
                print(f"自动识别目标列: {target_column}")
            else:
                raise ValueError(f"找不到目标列: {target_column}")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.feature_names = list(X.columns)
        self.target_name = target_column

        # Data validation
        print(f"\n数据质量检查:")
        print(f"- 缺失值: {X.isnull().sum().sum()}")
        print(f"- 目标变量缺失值: {y.isnull().sum()}")
        print(f"- 特征数量: {X.shape[1]}")
        print(f"- 样本数量: {X.shape[0]}")

        return X, y

    def preprocess_data(self, X, y, handle_missing='auto', handle_outliers='iqr'):
        """
        Preprocess data with missing value handling and outlier detection

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            handle_missing (str): How to handle missing values
            handle_outliers (str): How to handle outliers

        Returns:
            tuple: (X_processed, y_processed)
        """
        print("\n=== 数据预处理 ===")

        X_processed = X.copy()
        y_processed = y.copy()

        # Handle missing values
        if handle_missing == 'auto':
            # Numerical columns: median imputation
            numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if X_processed[col].isnull().sum() > 0:
                    median_val = X_processed[col].median()
                    X_processed[col].fillna(median_val, inplace=True)
                    print(f"- {col}: 用中位数 {median_val:.2f} 填充 {X_processed[col].isnull().sum()} 个缺失值")

            # Categorical columns: mode imputation
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X_processed[col].isnull().sum() > 0:
                    mode_val = X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'Unknown'
                    X_processed[col].fillna(mode_val, inplace=True)
                    print(f"- {col}: 用众数 '{mode_val}' 填充 {X_processed[col].isnull().sum()} 个缺失值")

        # Remove rows with missing target
        missing_target = y_processed.isnull().sum()
        if missing_target > 0:
            mask = ~y_processed.isnull()
            X_processed = X_processed[mask]
            y_processed = y_processed[mask]
            print(f"- 移除 {missing_target} 行目标变量缺失值")

        # Handle outliers in target variable
        if handle_outliers == 'iqr' and y_processed.dtype in [np.number, 'int64', 'float64']:
            Q1 = y_processed.quantile(0.25)
            Q3 = y_processed.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (y_processed >= lower_bound) & (y_processed <= upper_bound)
            outliers_removed = len(y_processed) - outlier_mask.sum()

            if outliers_removed > 0:
                X_processed = X_processed[outlier_mask]
                y_processed = y_processed[outlier_mask]
                print(f"- 移除 {outliers_removed} 个异常值 (IQR方法)")

        print(f"预处理后数据形状: {X_processed.shape}")
        return X_processed, y_processed

    def encode_categorical_features(self, X):
        """
        Encode categorical features to numerical

        Args:
            X (pd.DataFrame): Features with categorical columns

        Returns:
            pd.DataFrame: Features with encoded categorical variables
        """
        print("\n=== 分类特征编码 ===")

        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            print(f"发现 {len(categorical_cols)} 个分类特征:")

            label_encoders = {}

            for col in categorical_cols:
                # Use label encoding for high-cardinality features
                if X_encoded[col].nunique() > 10:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    label_encoders[col] = le
                    print(f"- {col}: 标签编码 ({X_encoded[col].nunique()} 个类别)")
                else:
                    # Use one-hot encoding for low-cardinality features
                    dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
                    X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
                    print(f"- {col}: 独热编码 ({X_encoded[col].nunique()} 个类别)")

            self.label_encoders = label_encoders

        self.feature_names = list(X_encoded.columns)
        print(f"编码后特征数量: {len(self.feature_names)}")

        return X_encoded

    def create_rfm_features(self, df, user_id_col='用户码', date_col='消费日期',
                           amount_col='总价', period_days=90):
        """
        Create RFM (Recency, Frequency, Monetary) features from transaction data

        Args:
            df (pd.DataFrame): Transaction data
            user_id_col (str): User identifier column
            date_col (str): Date column
            amount_col (str): Amount column
            period_days (int): Analysis period in days

        Returns:
            pd.DataFrame: RFM features
        """
        print(f"\n=== RFM特征工程 (分析期间: {period_days}天) ===")

        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Calculate analysis period
        max_date = df[date_col].max()
        period_start = max_date - pd.Timedelta(days=period_days)

        # Filter data for analysis period
        period_df = df[df[date_col] > period_start]

        # Calculate RFM metrics
        rfm_features = []

        for user_id in df[user_id_col].unique():
            user_data = period_df[period_df[user_id_col] == user_id]

            # Recency: Days since last purchase
            last_purchase = user_data[date_col].max()
            recency = (max_date - last_purchase).days

            # Frequency: Number of purchases
            frequency = len(user_data)

            # Monetary: Total spending
            monetary = user_data[amount_col].sum()

            # Additional features
            avg_order_value = monetary / frequency if frequency > 0 else 0
            days_since_first = (max_date - user_data[date_col].min()).days
            purchase_frequency = frequency / (days_since_first / 30) if days_since_first > 0 else 0

            rfm_features.append({
                user_id_col: user_id,
                'R_最近购买天数': recency,
                'F_购买频次': frequency,
                'M_总消费金额': monetary,
                'AOV_平均客单价': avg_order_value,
                'PF_购买频率': purchase_frequency,
                'Customer_客户活跃天数': days_since_first
            })

        rfm_df = pd.DataFrame(rfm_features)
        print(f"为 {len(rfm_df)} 个用户生成 RFM 特征")

        return rfm_df

    def create_interaction_features(self, X):
        """
        Create interaction features for improved model performance

        Args:
            X (pd.DataFrame): Original features

        Returns:
            pd.DataFrame: Features with interactions
        """
        print("\n=== 交互特征生成 ===")

        X_interactions = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        # Create pairwise interactions for top correlated features
        if len(numerical_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = X[numerical_cols].corr().abs()

            # Find top correlated pairs (excluding self-correlation)
            top_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.3:  # Threshold for creating interaction
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        top_pairs.append((col1, col2, corr_val))

            # Sort by correlation and take top 5
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = top_pairs[:5]

            for col1, col2, corr in top_pairs:
                interaction_name = f"{col1}_x_{col2}"
                X_interactions[interaction_name] = X[col1] * X[col2]
                print(f"- 创建交互特征: {interaction_name} (相关系数: {corr:.3f})")

        # Create polynomial features for important numerical features
        if len(numerical_cols) > 0:
            # Select features with highest variance
            variances = X[numerical_cols].var()
            top_variance_features = variances.nlargest(3).index

            for col in top_variance_features:
                squared_name = f"{col}_squared"
                X_interactions[squared_name] = X[col] ** 2
                print(f"- 创建平方特征: {squared_name}")

        print(f"交互特征生成完成，总特征数: {X_interactions.shape[1]}")
        return X_interactions

    def train_models(self, X, y, test_size=0.2, cv_folds=5):
        """
        Train multiple regression models and compare performance

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set proportion
            cv_folds (int): Cross-validation folds

        Returns:
            dict: Model performance results
        """
        print("\n=== 模型训练与评估 ===")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=1.0, random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state)
        }

        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        for name, model in models.items():
            print(f"\n--- {name} ---")

            # Choose appropriate data scaling
            if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            # Train model
            model.fit(X_tr, y_train)

            # Make predictions
            y_train_pred = model.predict(X_tr)
            y_test_pred = model.predict(X_te)

            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, y_train, cv=cv_folds, scoring='r2')

            # Store results
            results[name] = {
                'model': model,
                'scaler': scaler if 'Linear' in name or 'Ridge' in name or 'Lasso' in name else None,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_test_pred,
                'y_test': y_test
            }

            # Store model and scaler
            self.models[name] = model
            if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
                self.scalers[name] = scaler

            # Print results
            print(f"训练集 R²: {train_r2:.4f}")
            print(f"测试集 R²: {test_r2:.4f}")
            print(f"测试集 MAE: {test_mae:.4f}")
            print(f"测试集 RMSE: {test_rmse:.4f}")
            print(f"交叉验证 R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Select best model based on test R²
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name

        print(f"\n🏆 最佳模型: {best_model_name} (测试集 R²: {results[best_model_name]['test_r2']:.4f})")

        self.results = results
        return results

    def get_feature_importance(self, model_name=None, top_n=10):
        """
        Get feature importance from trained models

        Args:
            model_name (str): Name of the model (use best model if None)
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance with rankings
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in trained models")

        model = self.models[model_name]

        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        else:
            raise ValueError(f"Model '{model_name}' does not support feature importance")

        # Create importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })

        # Sort and rank
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance['rank'] = range(1, len(feature_importance) + 1)
        feature_importance['importance_pct'] = (feature_importance['importance'] /
                                               feature_importance['importance'].sum() * 100)

        return feature_importance.head(top_n)

    def predict(self, X, model_name=None):
        """
        Make predictions using trained model

        Args:
            X (pd.DataFrame): Features for prediction
            model_name (str): Model to use (best model if None)

        Returns:
            np.array: Predictions
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in trained models")

        model = self.models[model_name]

        # Apply scaling if needed
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
            return model.predict(X_scaled)
        else:
            return model.predict(X)

    def run_complete_analysis(self, file_path, target_column,
                            create_rfm=False, create_interactions=False,
                            rfm_config=None, **kwargs):
        """
        Run complete regression analysis pipeline

        Args:
            file_path (str): Path to data file
            target_column (str): Target variable column
            create_rfm (bool): Whether to create RFM features
            create_interactions (bool): Whether to create interaction features
            rfm_config (dict): RFM configuration
            **kwargs: Additional parameters

        Returns:
            dict: Complete analysis results
        """
        print("🚀 开始完整回归分析")
        print("=" * 50)

        # 1. Load and validate data
        X, y = self.load_and_validate_data(file_path, target_column, **kwargs)

        # 2. Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y)

        # 3. Handle categorical features
        X_encoded = self.encode_categorical_features(X_processed)

        # 4. Create RFM features if requested
        if create_rfm:
            if rfm_config is None:
                rfm_config = {
                    'user_id_col': '用户码',
                    'date_col': '消费日期',
                    'amount_col': '总价'
                }
            # This would need the original transaction data
            # For now, we'll skip RFM feature creation in the main pipeline
            pass

        # 5. Create interaction features if requested
        if create_interactions:
            X_final = self.create_interaction_features(X_encoded)
        else:
            X_final = X_encoded

        # 6. Train models
        results = self.train_models(X_final, y_processed)

        # 7. Get feature importance
        feature_importance = self.get_feature_importance()

        # 8. Create summary
        summary = {
            'data_shape': X_final.shape,
            'feature_count': X_final.shape[1],
            'sample_count': len(y_processed),
            'best_model': self.best_model_name,
            'best_r2': results[self.best_model_name]['test_r2'],
            'best_mae': results[self.best_model_name]['test_mae'],
            'feature_importance': feature_importance
        }

        print(f"\n✅ 分析完成！")
        print(f"最佳模型: {self.best_model_name}")
        print(f"测试集 R²: {summary['best_r2']:.4f}")
        print(f"测试集 MAE: {summary['best_mae']:.4f}")

        return {
            'results': results,
            'summary': summary,
            'X_final': X_final,
            'y_final': y_processed,
            'feature_importance': feature_importance
        }

def main():
    """Example usage"""
    analyzer = RegressionAnalyzer()

    # Example with housing price data
    file_path = "房价预测数据.csv"
    target_column = "房价"

    if pd.io.common.file_exists(file_path):
        analysis_results = analyzer.run_complete_analysis(
            file_path,
            target_column,
            create_interactions=True
        )

        print(f"\n特征重要性排名:")
        print(analysis_results['feature_importance'])

    else:
        print(f"数据文件未找到: {file_path}")
        print("请提供正确的CSV数据文件路径")

if __name__ == "__main__":
    main()