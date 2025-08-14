# QWEN.md

## 项目概述

这是一个使用RFM（最近购买时间Recency、购买频率Frequency、消费金额Monetary）模型对电商平台用户进行聚类分析的项目。项目的主要目标是根据用户的历史购买行为，将用户划分为不同的价值群体，以便进行针对性的市场营销和客户关系管理。

核心分析在 Jupyter Notebook (`RFM分析.ipynb`) 中完成，使用 Python 和相关的数据科学库（如 pandas, scikit-learn）来处理数据、计算 RFM 指标、执行 K-Means 聚类，并对结果进行可视化。项目还包含一个 Python 脚本 (`customer_segmentation_report.py`)，用于根据聚类结果生成详细的分析报告、统计数据和可视化图表。

## 关键文件

- **`RFM分析.ipynb`**: 包含完整 RFM 分析流程的 Jupyter Notebook。涵盖了从数据加载、清洗、RFM 指标计算、K-Means 聚类到结果可视化的所有步骤。
- **`电商历史订单.csv`**: 包含电商平台历史交易数据的原始 CSV 文件。主要字段包括订单号、产品码、消费日期、产品说明、数量、单价、用户码、城市。
- **`customer_segments_clean.csv`**: 经过 RFM 分析和聚类后生成的用户分群结果文件。包含用户码、R/F/M 值、聚类得分和最终的客户价值分类（高、中、低）。
- **`customer_segmentation_report.py`**: 用于生成分析报告、统计摘要和可视化图表的 Python 脚本。它读取原始数据和分群结果，并生成多个输出文件。
- **`data_validation_report.txt`**: 由 `customer_segmentation_report.py` 生成的数据验证报告，总结了数据质量、分群分布和基本统计信息。
- **`segment_summary_statistics.csv`**: 由 `customer_segmentation_report.py` 生成的按分群统计的详细指标摘要。
- **`customer_segmentation_dashboard.png`**: 由 `customer_segmentation_report.py` 生成的关键分析结果可视化仪表板。
- **`vip_customers_list.csv`**: 由 `customer_segmentation_report.py` 生成的高价值客户清单，用于营销活动。
- **`executive_summary.txt`**: 由 `customer_segmentation_report.py` 生成的项目执行摘要，提供核心发现和营销建议。

## 依赖项

- Python 3.x
- Jupyter Notebook (用于执行 `.ipynb` 文件)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 使用方法

1.  **环境准备**:
    *   确保安装了 Python 3.x。
    *   安装必要的 Python 库，可以通过 `pip install pandas numpy matplotlib seaborn scikit-learn jupyter` 命令安装。
    *   确保系统支持中文显示（如代码中设置的 `SimHei` 字体）。

2.  **执行分析**:
    *   打开 Jupyter Notebook 环境。
    *   运行 `RFM分析.ipynb` 中的所有单元格，以执行完整的 RFM 分析流程并生成 `customer_segments_clean.csv` 文件。

3.  **生成报告**:
    *   在命令行中执行 `customer_segmentation_report.py` 脚本：
        ```bash
        python customer_segmentation_report.py
        ```
    *   脚本将读取 `电商历史订单.csv` 和 `customer_segments_clean.csv`，并生成以下输出文件：
        *   `data_validation_report.txt`
        *   `segment_summary_statistics.csv`
        *   `customer_segmentation_dashboard.png`
        *   `vip_customers_list.csv`
        *   `executive_summary.txt`

## 数据处理流程

1.  **数据加载**: 从 `电商历史订单.csv` 读取原始交易数据。
2.  **数据清洗**: 移除数量为负数的异常订单记录。
3.  **RFM 计算**:
    *   **R (Recency)**: 计算每个用户自最近一次购买以来的天数。
    *   **F (Frequency)**: 统计每个用户的总购买次数。
    *   **M (Monetary)**: 计算每个用户的总消费金额。
4.  **聚类分析**: 对 R、F、M 三个维度分别使用 K-Means 聚类算法进行分组（通常分为 3-4 层）。
5.  **聚类排序**: 根据 R/F/M 的实际值对聚类结果进行排序，确保聚类标签具有业务含义（例如，高价值对应高 M 值）。
6.  **客户价值评分**: 将 R、F、M 三个维度的聚类层级得分相加，得到每个用户的总分。
7.  **客户价值分类**: 根据总分将用户划分为“高价值”、“中价值”和“低价值”三类。
8.  **结果输出**: 将用户码、R/F/M 值、总分和客户价值分类保存到 `customer_segments_clean.csv`。