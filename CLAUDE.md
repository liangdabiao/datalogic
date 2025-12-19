# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive data analysis educational repository based on "数据分析咖哥十话" (Data Analysis Coffee Talk) book series. The project contains multiple data science modules covering fundamental analysis techniques used in business intelligence and e-commerce analytics.

## Architecture & Project Structure

The repository is organized into thematic modules, each focusing on specific data analysis techniques:

```
datalogic-main/
├── 01 用户画像/      # User profiling & segmentation
├── 02 聚类分析/      # Customer clustering with RFM analysis
├── 03 回归分析/      # Regression analysis for LTV prediction
├── 04 归因分析/      # Marketing channel attribution
├── 05 漏斗模型/      # Conversion funnel analysis
├── 06 留存分析/      # Customer retention analysis
├── 07 内容分析/      # Content analysis with LLM integration
├── 08 推荐系统/      # Recommendation systems (collaborative filtering, SVD)
├── 09 AB测试/        # A/B testing and statistical analysis
├── 10 增长模型/      # Growth modeling & uplift analysis
├── 第2课_数据探索和数据可视化/  # Data visualization fundamentals
├── 第3课_回归算法与生命周期价值预测/  # Advanced regression techniques
└── linuxdo/          # Community data analysis case study
```

## Key Technologies & Dependencies

### Core Python Libraries
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn (Chinese font: SimHei)
- **Statistical Analysis**: scipy
- **Jupyter Notebook**: Interactive analysis environment

### Specific Module Dependencies
- **Content Analysis (07)**: LLM integration for text analysis
- **Recommendation Systems (08)**: Collaborative filtering algorithms
- **Growth Modeling (10)**: XGBoost uplift modeling, Qini curve analysis

## Common Development Commands

### Running Analysis Notebooks
```bash
# Launch Jupyter for interactive analysis
jupyter notebook

# Execute notebooks programmatically
jupyter nbconvert --to notebook --inplace --execute [notebook_name].ipynb

# Example: Run RFM analysis
jupyter nbconvert --to notebook --inplace --execute "02 聚类分析/RFM分析.ipynb"
```

### Running Python Analysis Scripts
```bash
# Growth model analysis (order of execution matters)
cd "10 增长模型"
python "01_裂变策略效果评估/full_analysis_fixed.py"
python "02_用户细分与个性化策略/user_segmentation_analysis.py"
python "05_增长建模_xgboost_qini/xgboost_uplift_modeling.py"
python "05_增长建模_xgboost_qini/qini_curve_analysis.py"

# Recommendation system evaluation
cd "08 推荐系统"
python model_evaluation_v2.py

# User profiling analysis
cd "01 用户画像"
python 用户分群分析.py
```

## Data Analysis Patterns

### Standard Analysis Workflow
1. **Data Loading**: CSV files with Chinese e-commerce data
2. **Preprocessing**: Data cleaning, feature engineering, date parsing
3. **Exploratory Analysis**: Statistical summaries, visualization
4. **Model Implementation**: Apply domain-specific algorithms
5. **Evaluation**: Performance metrics, visualization of results
6. **Reporting**: Generate markdown and HTML reports

### Common Data Schemas
- **Customer Data**: 用户码, 城市, 订单金额, 消费日期
- **Product Data**: 产品码, 产品说明, 数量, 单价
- **RFM Analysis**: Recency, Frequency, Monetary values
- **Growth Modeling**: 裂变策略, 转化率, 增量价值

## Module-Specific Guidelines

### Clustering Analysis (02)
- Uses K-means for customer segmentation
- RFM (Recency, Frequency, Monetary) modeling approach
- Elbow method for optimal cluster determination

### Regression Analysis (03)
- Linear regression for LTV (Lifetime Value) prediction
- Housing price prediction dataset for regression practice
- Model evaluation using R² scores

### Growth Modeling (10)
- **Most complex module** - follow execution order carefully
- XGBoost uplift modeling for causal inference
- Qini curve analysis for treatment effect evaluation
- ROI analysis across different marketing strategies

### Recommendation Systems (08)
- Item-based collaborative filtering
- SVD (Singular Value Decomposition) matrix factorization
- Model evaluation with MAE/RMSE metrics

## Important Notes

- All analysis modules use Chinese language throughout (data, comments, reports)
- Chinese font support (SimHei) required for proper visualization
- Each module contains both Jupyter notebooks and Python scripts
- Many modules generate HTML reports alongside analysis results
- Data files are typically CSV format with Chinese field names
- Follow the numbered order within modules for progressive learning path

## File Organization Patterns

Most modules follow this structure:
- `.ipynb` files: Interactive Jupyter analysis notebooks
- `*.py` files: Standalone analysis scripts
- `analysis_plan.md`: Detailed analysis methodology
- `analysis_report.md/html`: Generated analysis reports
- `*.csv`: Intermediate data processing results
- `*.png`: Generated visualizations and charts