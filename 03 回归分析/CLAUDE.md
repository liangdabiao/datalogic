# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a regression analysis project from "数据分析咖哥十话" (Data Analysis Coffee Talk) book series, focusing on customer lifetime value (LTV) prediction using Chinese e-commerce data and regression analysis techniques.

## Architecture & Data Structure
- **Primary Notebook**: `预测LTV.ipynb` - Jupyter notebook for customer LTV prediction
- **Data Sources**: 
  - `电商历史订单.csv` - E-commerce historical order data (87,180+ rows)
  - `房价预测数据.csv` - Housing price prediction dataset for regression analysis
- **Key Features**: RFM analysis (Recency, Frequency, Monetary) for LTV prediction

## Development Setup
### Required Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning models (Linear Regression)
- matplotlib: Data visualization (Chinese font: SimHei)

### Common Commands
- **Run Jupyter Notebook**: `jupyter notebook 预测LTV.ipynb`
- **Install Dependencies**: `pip install pandas numpy scikit-learn matplotlib`
- **View Data**: `python -c "import pandas as pd; print(pd.read_csv('房价预测数据.csv').head())"`

## Key Analysis Flow
1. **Data Preprocessing**: Load CSV, date parsing, calculate order totals
2. **RFM Modeling**: Calculate R (recency), F (frequency), M (monetary) values
3. **Feature Engineering**: Create training features from 3-month window
4. **Model Training**: Linear regression on LTV prediction
5. **Evaluation**: R² score comparison between training vs test sets

## Data Schema
**电商历史订单.csv**:
- 订单号, 产品码, 消费日期, 产品说明, 数量, 单价, 用户码, 城市
- Date range: 2022-06-01 to 2023-06-09

**房价预测数据.csv**:
- 房屋ID, 面积, 房间数, 卫生间数, 楼层, 总楼层, 建造年份, 地铁距离, 学校距离, 商场距离, 装修等级, 朝向, 小区类型, 房价