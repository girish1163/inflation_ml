# Inflation Prediction ML Model

A comprehensive machine learning project for predicting inflation rates using multiple algorithms with hyperparameter tuning and advanced visualizations.

## 📊 Project Overview

This project implements and compares various machine learning models to predict inflation rates for India. The models are trained with hyperparameter optimization and include detailed performance analysis with visualizations.

## 🚀 Features

- **Multiple ML Models**: Random Forest, XGBoost, Gradient Boosting, SVR, Neural Network, and Stacking Ensemble
- **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit cross-validation
- **Advanced Feature Engineering**: Lag features, rolling statistics, and time-based features
- **Comprehensive Visualization**: 8+ detailed plots including model comparison, feature importance, residual analysis, and forecasting
- **Bootstrapped Confidence Intervals**: 5-year forecast with uncertainty quantification
- **Non-interactive Mode**: All plots saved as high-resolution PNG files

## 📁 Project Structure

```
windsurf-project-7/
├── inflation_ml_model.py    # Main script
├── inflation_data.csv        # Dataset (required)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── model_results.csv         # Model performance metrics
├── inflation_forecast.csv    # 5-year forecast values
└── *.png                     # Generated visualizations
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd windsurf-project-7
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Requirements

The project requires the following Python packages:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0

## 🏃‍♂️ Usage

1. Ensure `inflation_data.csv` is in the project directory
2. Run the main script:
```bash
python inflation_ml_model.py
```

## 📈 Output

The script generates the following files:

### Model Performance
- `model_results.csv`: Detailed performance metrics for all models
- `model_comparison.png`: Visual comparison of model performance

### Visualizations
- `feature_importance.png`: Feature importance for tree-based models
- `correlation_heatmap.png`: Feature correlation matrix
- `feature_pairplot.png`: Pairplot of top correlated features
- `time_series_decomposition.png`: Decomposition of time series
- `residual_analysis.png`: Analysis of model residuals
- `rolling_rmse.png`: Model performance over time

### Forecast
- `inflation_forecast.png`: 5-year forecast with confidence intervals
- `inflation_forecast.csv`: Forecast values with 95% confidence intervals

## 🎯 Model Performance

Based on the latest run:

| Model | RMSE | R² | MAE | MAPE (%) |
|-------|------|----|-----|----------|
| SVR | 0.83 | 0.42 | 0.75 | 16.24 |
| Random Forest | 0.99 | 0.18 | 0.81 | 17.68 |
| Gradient Boosting | 1.07 | 0.05 | 0.85 | 19.76 |
| XGBoost | 1.25 | -0.31 | 1.15 | 25.85 |
| Neural Network | 2.31 | -3.47 | 2.12 | 47.96 |
| Stacking | 2.74 | -5.29 | 2.08 | 51.41 |

**Best Model**: SVR (Support Vector Regression)

## 🔮 5-Year Forecast (2025-2029)

The best model (SVR) predicts the following inflation rates for India:

| Year | Forecast (%) | Lower 95% | Upper 95% |
|------|--------------|-----------|-----------|
| 2025 | ~5.2 | ~4.1 | ~7.2 |
| 2026 | ~4.7 | ~3.1 | ~7.0 |
| 2027 | ~5.1 | ~2.7 | ~8.2 |
| 2028 | ~4.8 | ~1.9 | ~8.5 |
| 2029 | ~5.0 | ~1.4 | ~9.2 |

## 📊 Data Format

The `inflation_data.csv` should contain:
- `country_name`: Country names
- `indicator_name`: Indicator descriptions
- Year columns (e.g., 2010, 2011, ...): Inflation values for each year

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 Technical Details

### Feature Engineering
- **Lag Features**: 1-4 period lags
- **Rolling Statistics**: Mean, std, min, max for 3, 5, 7 period windows
- **Time Features**: Sine/cosine transformations for cyclical patterns
- **Yearly Statistics**: Year-wise average inflation

### Model Training
- **Cross-Validation**: TimeSeriesSplit with 5 folds
- **Hyperparameter Tuning**: GridSearchCV for each model
- **Evaluation Metrics**: MAE, RMSE, R², MAPE

### Forecasting
- **Method**: Bootstrapping with 100 samples
- **Confidence Intervals**: 95% confidence bands
- **Horizon**: 5-year forecast

## 🐛 Issues

If you encounter any issues:
1. Check that all dependencies are installed
2. Ensure `inflation_data.csv` is present and correctly formatted
3. Verify Python version (recommended: 3.8+)

## 📧 Contact

[Your Name] - [Your Email] - [Your GitHub Profile]

---

⭐ If you find this project useful, please give it a star!
