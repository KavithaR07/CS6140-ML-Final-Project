# Market-Regime-Aware Next-Day Stock Prediction

**Course:** CS6140 – Machine Learning, Fall 2026  
**University:** Northeastern University  
**Team Members:** Kavitha and Julian

## a. Project Overview

This project investigates whether market regimes identified through unsupervised learning can improve next-day stock return predictions.

We aim to combine company-specific financial reports, historical stock prices, and market-wide economic indicators into a company-day dataset. This dataset will include both cross-sectional and temporal information, rather than exclusively time-series data.

## b. Research Questions

1. Do market-wide indicators improve next-day stock predictions beyond company-specific features?
2. Does incorporating market regimes identified through Gaussian Mixture Models (GMM) improve prediction performance?
3. How do Ridge Regression, Random Forest, and AdaBoost compare across different market conditions?

## c. Proposed Data Sources

- **SEC EDGAR:** Company financial statements, revenue growth, profitability, and leverage.
- **Yahoo Finance (yfinance):** Historical daily stock prices, trading volume, and market returns.
- **FRED:** Economic indicators, including market volatility and interest rates.

Each observation will represent an individual company on a specific trading day, incorporating company-specific financial characteristics and market-wide conditions.

## d. Proposed Machine Learning Methods

- **Gaussian Mixture Models (GMM):** Identify market regimes using unsupervised clustering.
- **Ridge Regression:** Establish a regularized linear regression benchmark.
- **Random Forest:** Capture nonlinear relationships between financial features and stock returns.
- **AdaBoost:** Investigate boosting performance and sensitivity to noisy financial data.
- **Walk-Forward Validation:** Evaluate predictions using chronological training and testing periods.

## e. Team Responsibilities

### Kavitha
- GitHub repository management and organization.
- Data collection support.
- Data Preprocessing and validation
- Exploratory Data Analysis
- Random Forest implementation.
- AdaBoost implementation.

### Julian
- Data collection and integration.
- Feature engineering.
- GMM market-regime clustering.
- Financial reporting schema alignment.
- Ridge Regression implementation.

### Shared Responsibilities — Kavitha and Julian
- Finalizing and reviewing feature engineering.
- Implementing and testing walk-forward validation.
- Model evaluation and performance comparison.
- Analyzing and interpreting results.
- Preparing the final report.
- Preparing and delivering the final presentation.

## f. Project Workflow

1. Collect historical stock prices, financial reports, and economic indicators.
2. Integrate datasets into a company-day structure.
3. Clean the dataset and perform exploratory data analysis.
4. Develop and select relevant financial features.
5. Identify market regimes using GMM clustering.
6. Train Ridge Regression, Random Forest, and AdaBoost models.
7. Perform walk-forward validation.
8. Evaluate and compare model performance across market regimes.
9. Visualize results and discuss findings.
10. Complete the final report and presentation.

## g. Repository Structure

- `data/` — Collected and processed datasets.
- `notebooks/` — Exploratory data analysis and ML experiments.
- `src/` — Python scripts for data processing and modeling.
- `results/` — Model evaluation results, metrics, and visualizations.
- `docs/` — Project proposal, methodology, and final report.
- `requirements.txt` — Python dependencies.

## h. Current Project Status

The GitHub repository structure and project documentation are being prepared. Dataset collection, feature engineering, model implementation, and evaluation are planned but have not yet been completed.
