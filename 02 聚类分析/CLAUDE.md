# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a customer segmentation analysis project using RFM (Recency, Frequency, Monetary) clustering on e-commerce transaction data. The project contains Jupyter notebooks implementing unsupervised learning techniques to identify high-value customers for targeted marketing.

## Key Files

- `RFM分析.ipynb`: Main analysis notebook containing:
  - Data loading and cleaning for order history
  - RFM metric calculation (Recency, Frequency, Monetary)
  - K-means clustering for customer segmentation
  - Visualization of customer segments
  - VIP customer identification logic

- `电商历史订单.csv`: Transaction data with fields:
  - 订单号 (order_id), 产品码 (product_code), 消费日期 (purchase_date)
  - 产品说明 (product_description), 数量 (quantity), 单价 (unit_price)
  - 用户码 (user_id), 城市 (city)

- `电商用户数据.csv`: Customer data with fields:
  - 客户ID (customer_id), 订单日期 (order_date), 订单金额 (order_amount), 商品类别 (product_category)

## Dependencies

- Python 3.x with pandas, numpy, matplotlib, seaborn, scikit-learn
- Jupyter notebook environment
- Chinese font support (SimHei) for proper display

## Usage

```bash
# Run the analysis
jupyter notebook RFM分析.ipynb

# Alternative execution via command line
jupyter nbconvert --to notebook --inplace --execute RFM分析.ipynb

# Save cleaned results
df_user.to_csv('customer_segments.csv', index=False, encoding='utf-8-sig')
```

## Data Processing Pipeline

1. **Data Loading**: Read historical order data from CSV
2. **Data Cleaning**: Remove orders with negative quantities
3. **RFM Calculation**: 
   - R (Recency): Days since last purchase
   - F (Frequency): Number of purchases
   - M (Monetary): Total purchase amount
4. **Clustering**: K-Means clustering on each RFM dimension
5. **Scoring**: Combine clusters to create customer segments
6. **Visualization**: 2D and 3D scatter plots showing segments

## Key Custom Functions

- `show_elbow(df)`: Plots elbow method for optimal k determination
- `order_cluster()`: Orders clusters by mean values for consistent ranking
- Customer value classification: "低价值" (Low), "中价值" (Medium), "高价值" (High)

## Notes

- Analysis includes Chinese documentation and output
- Customer segments are automatically labeled for VIP identification
- All Chinese field names are preserved in data processing