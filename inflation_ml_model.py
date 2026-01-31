import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    StackingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error
)
from sklearn.model_selection import (
    TimeSeriesSplit, 
    GridSearchCV,
    cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend for saving plots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for better-looking plots
try:
    # Try seaborn style first (for older matplotlib versions)
    plt.style.use('seaborn-v0_8')
except:
    try:
        # Fallback to default style with seaborn color palette
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
    except Exception as e:
        print(f"Warning: Could not set plot style: {e}")

# Set color palette
sns.set_palette('viridis')

# ----------------------------
# Load and Preprocess Data
# ----------------------------
print("Loading data...")
df = pd.read_csv("inflation_data.csv")
df_long = df.melt(
    id_vars=["country_name", "indicator_name"],
    var_name="Year",
    value_name="Inflation"
)
df_long["Year"] = df_long["Year"].astype(int)
df_long = df_long.dropna()

# Feature Engineering
def create_features(df):
    df = df.copy()
    df['Year'] = pd.to_numeric(df['Year'])
    
    # Lag features
    for lag in [1, 2, 3, 4]:
        df[f'Inflation_lag{lag}'] = df['Inflation'].shift(lag)
    
    # Rolling statistics
    windows = [3, 5, 7]
    for window in windows:
        df[f'rolling_mean_{window}'] = df['Inflation'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Inflation'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['Inflation'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['Inflation'].rolling(window=window).max()
    
    # Yearly statistics
    df['yearly_avg'] = df.groupby('Year')['Inflation'].transform('mean')
    
    # Time-based features
    df['year_sin'] = np.sin(2 * np.pi * (df['Year'] - df['Year'].min()) / 10)
    df['year_cos'] = np.cos(2 * np.pi * (df['Year'] - df['Year'].min()) / 10)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

# Prepare data for modeling
print("Preparing data for modeling...")

# Select country and prepare data
country = "India"
print(f"Preparing data for {country}...")

# Get data for selected country and sort by year
data = df_long[df_long["country_name"] == country].sort_values('Year')

# Ensure we have the required columns
if 'Inflation' not in data.columns:
    # Try to find the inflation column if it has a different name
    possible_inflation_cols = [col for col in data.columns if 'inflation' in col.lower() or 'cpi' in col.lower()]
    if possible_inflation_cols:
        data = data.rename(columns={possible_inflation_cols[0]: 'Inflation'})
    else:
        raise ValueError("Could not find inflation data column in the dataset")

# Apply feature engineering
data = create_features(data)

# Verify we have enough data
if len(data) < 10:  # arbitrary minimum number of data points
    raise ValueError(f"Not enough data points for {country}. Found only {len(data)} records.")

# Ensure all required features are present
required_features = ['Year', 'Inflation']
for col in required_features:
    if col not in data.columns:
        raise ValueError(f"Required column '{col}' not found in data")

# Create features if they don't exist
if 'Inflation_lag1' not in data.columns:
    data['Inflation_lag1'] = data['Inflation'].shift(1)

# Calculate rolling statistics if they don't exist
for window in [3, 5, 7]:
    if f'rolling_mean_{window}' not in data.columns:
        data[f'rolling_mean_{window}'] = data['Inflation'].rolling(window=window).mean()
    if f'rolling_std_{window}' not in data.columns:
        data[f'rolling_std_{window}'] = data['Inflation'].rolling(window=window).std()

# Drop rows with NaN values that were created by lag/rolling operations
data = data.dropna()

# Select features for modeling
feature_columns = ['Year', 'Inflation_lag1', 'rolling_mean_3', 'rolling_std_3']
X = data[feature_columns]
y = data['Inflation']

# Split into train and test sets (80-20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# ----------------------------
# Model Definitions with Hyperparameter Grids
# ----------------------------
print("\nInitializing models with hyperparameter grids...")

# Base models with parameter grids for tuning
models = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'SVR': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ]),
        'params': {
            'svr__kernel': ['rbf', 'linear'],
            'svr__C': [0.1, 1, 10],
            'svr__gamma': ['scale', 'auto'],
            'svr__epsilon': [0.01, 0.1, 0.5]
        }
    },
    'Neural Network': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                activation='relu',
                solver='adam',
                random_state=42,
                max_iter=1000,
                early_stopping=True
            ))
        ]),
        'params': {
            'mlp__hidden_layer_sizes': [(50, 25), (100, 50)],
            'mlp__alpha': [0.0001, 0.001],
            'mlp__learning_rate_init': [0.001, 0.01],
            'mlp__batch_size': [16, 32]
        }
    }
}

# ----------------------------
# Model Training and Evaluation with Cross-Validation
# ----------------------------
results = {}
tscv = TimeSeriesSplit(n_splits=5)

for name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"Training and tuning {name}...")
    print(f"{'='*50}")
    
    try:
        # Perform grid search with time series cross-validation
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model and predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Cross-validated scores
        cv_scores = cross_val_score(
            best_model, 
            X_train, 
            y_train, 
            cv=tscv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Calculate additional metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Store results with all metrics
        results[name] = {
            'model': best_model,
            'y_pred': y_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'cv_mean_rmse': cv_scores.mean(),
            'cv_std_rmse': cv_scores.std()
        }
        
        # Print model performance
            
        print(f"\n{name} - Performance:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  CV Mean RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
    except Exception as e:
        print(f"Error training {name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# ----------------------------
# Ensemble Model (Stacking)
# ----------------------------
print("\nTraining Stacking Ensemble...")
try:
    # Get the best performing models for stacking
    best_models = []
    for name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        if name in results:
            best_models.append((name.lower().replace(' ', '_'), results[name]['model']))
    
    if len(best_models) >= 2:  # Need at least 2 models for stacking
        stacking_model = StackingRegressor(
            estimators=best_models,
            final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            n_jobs=-1
        )
        
        stacking_model.fit(X_train, y_train)
        y_pred_stack = stacking_model.predict(X_test)
        
        # Calculate metrics for stacking
        mae_stack = mean_absolute_error(y_test, y_pred_stack)
        rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
        r2_stack = r2_score(y_test, y_pred_stack)
        mape_stack = mean_absolute_percentage_error(y_test, y_pred_stack) * 100
        
        # Store stacking results with all metrics
        results['Stacking'] = {
            'model': stacking_model,
            'y_pred': y_pred_stack,
            'mae': mae_stack,
            'rmse': rmse_stack,
            'r2': r2_stack,
            'mape': mape_stack,
            'cv_mean_rmse': np.nan,  # Not calculated for stacking
            'cv_std_rmse': np.nan    # Not calculated for stacking
        }
        print(f"Stacking - MAE: {results['Stacking']['mae']:.4f}, RMSE: {results['Stacking']['rmse']:.4f}, R²: {results['Stacking']['r2']:.4f}")
    else:
        print("Not enough models available for stacking")
except Exception as e:
    print(f"Error in stacking: {str(e)}")

# ----------------------------
# Results Comparison and Visualization
# ----------------------------
if results:
    # Create results dataframe
    results_df = pd.DataFrame({
        'Model': results.keys(),
        'MAE': [x['mae'] for x in results.values()],
        'RMSE': [x['rmse'] for x in results.values()],
        'R2': [x['r2'] for x in results.values()],
        'MAPE': [x['mape'] for x in results.values()],
        'CV_RMSE_Mean': [x['cv_mean_rmse'] for x in results.values()],
        'CV_RMSE_Std': [x['cv_std_rmse'] for x in results.values()]
    }).sort_values('RMSE')

    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))
    
    # Plot model comparison
    plt.figure(figsize=(15, 8))
    
    # RMSE comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='RMSE', data=results_df.sort_values('RMSE'))
    plt.title('Model RMSE Comparison (Lower is Better)')
    plt.xticks(rotation=45, ha='right')
    
    # R2 comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='R2', data=results_df.sort_values('R2', ascending=False))
    plt.title('Model R² Comparison (Higher is Better)')
    plt.xticks(rotation=45, ha='right')
    
    # MAPE comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='MAPE', data=results_df.sort_values('MAPE'))
    plt.title('Model MAPE % Comparison (Lower is Better)')
    plt.xticks(rotation=45, ha='right')
    
    # Cross-validation RMSE comparison
    plt.subplot(2, 2, 4)
    cv_data = results_df.dropna(subset=['CV_RMSE_Mean', 'CV_RMSE_Std'])
    if not cv_data.empty:
        ax = sns.barplot(x='Model', y='CV_RMSE_Mean', data=cv_data.sort_values('CV_RMSE_Mean'))
        
        # Add error bars manually to handle NaN values
        for i, (_, row) in enumerate(cv_data.sort_values('CV_RMSE_Mean').iterrows()):
            if not np.isnan(row['CV_RMSE_Std']):
                plt.errorbar(
                    x=i,
                    y=row['CV_RMSE_Mean'],
                    yerr=row['CV_RMSE_Std'],
                    color='black',
                    capsize=5,
                    fmt='none'
                )
        plt.title('Cross-Validated RMSE (Lower is Better)')
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, 'No valid cross-validation data', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Cross-Validated RMSE (No Data)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    results_df.to_csv('model_results.csv', index=False)
    print("\nSaved model comparison results to 'model_results.csv' and 'model_comparison.png'")
else:
    print("\nNo models were successfully trained.")

# ----------------------------
# Feature Analysis
# ----------------------------
if results:
    try:
        # Plot feature importance for tree-based models
        tree_based_models = [
            (name, result) for name, result in results.items()
            if hasattr(result['model'], 'feature_importances_') or 
               any(hasattr(step[1], 'feature_importances_') for step in 
                   getattr(result['model'], 'steps', []))
        ]
        
        if tree_based_models:
            plt.figure(figsize=(15, 5 * len(tree_based_models)))
            
            for i, (name, result) in enumerate(tree_based_models, 1):
                model = result['model']
                
                # Handle pipeline models
                if hasattr(model, 'named_steps'):
                    for step_name, step_model in model.named_steps.items():
                        if hasattr(step_model, 'feature_importances_'):
                            model = step_model
                            break
                
                if hasattr(model, 'feature_importances_'):
                    plt.subplot(len(tree_based_models), 1, i)
                    
                    # Get feature importances
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot
                    plt.title(f'{name} - Feature Importance', fontsize=12)
                    bars = plt.bar(range(len(importances)), importances[indices])
                    plt.xticks(range(len(importances)), 
                             [X.columns[i] for i in indices], 
                             rotation=45, ha='right')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.4f}',
                               ha='center', va='bottom', rotation=0, fontsize=8)
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = pd.concat([X, y], axis=1).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pairplot for top correlated features
        top_features = corr['Inflation'].abs().sort_values(ascending=False).index[1:6]
        if len(top_features) > 1:
            sns.pairplot(pd.concat([X[top_features], y], axis=1))
            plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
            plt.tight_layout()
            plt.savefig('feature_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"\nError in feature analysis: {str(e)}")
        import traceback
        traceback.print_exc()

# ----------------------------
# Time Series Analysis and Forecast
# ----------------------------
if results:
    # Get best model based on RMSE
    best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    best_model = results[best_model_name]['model']
    print(f"\n{'='*50}")
    print(f"Generating forecast using best model: {best_model_name}")
    print(f"{'='*50}")
    
    # 1. Time Series Decomposition
    try:
        print("\nPerforming time series decomposition...")
        # Ensure we have a regular time series
        ts_data = data.set_index('Year')['Inflation'].asfreq('YE')
        decomposition = seasonal_decompose(ts_data, period=5)  # 5-year cycle
        
        plt.figure(figsize=(14, 10))
        
        # Plot original series
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Observed')
        
        # Plot trend
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        
        # Plot seasonality
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal')
        
        # Plot residuals
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.axhline(0, color='r', linestyle='--', alpha=0.3)
        plt.title('Residuals')
        
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in time series decomposition: {str(e)}")
    
    # 2. Residual Analysis
    try:
        print("\nAnalyzing residuals...")
        y_pred = results[best_model_name]['y_pred']
        residuals = y_test - y_pred
        
        plt.figure(figsize=(15, 10))
        
        # Residuals vs Fitted
        plt.subplot(2, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Fitted')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        
        # QQ Plot
        plt.subplot(2, 2, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot')
        
        # Residuals histogram
        plt.subplot(2, 2, 3)
        sns.histplot(residuals, kde=True)
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        
        # Residuals over time
        plt.subplot(2, 2, 4)
        plt.plot(y_test.index, residuals, 'o', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals Over Time')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test for normality
        _, p_value = stats.normaltest(residuals)
        print(f"Normality test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Residuals do not follow a normal distribution (p < 0.05)")
    except Exception as e:
        print(f"Error in residual analysis: {str(e)}")
    
    # 3. Forecast with Confidence Intervals using Bootstrapping
    print("\nGenerating forecast with confidence intervals...")
    
    # Prepare future data
    last_data = X.iloc[-1:].copy()
    future_years = range(X['Year'].max() + 1, X['Year'].max() + 6)
    n_bootstraps = 100
    bootstrap_forecasts = np.zeros((n_bootstraps, len(future_years)))
    
    # Bootstrap forecast
    for i in range(n_bootstraps):
        # Sample with replacement
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]
        
        # Train on bootstrap sample
        model = clone(best_model)
        model.fit(X_boot, y_boot)
        
        # Forecast
        temp_data = last_data.copy()
        bootstrap_predictions = []
        for j, year in enumerate(future_years):
            temp_data['Year'] = year
            # Update lag features
            for lag in range(1, 5):
                if f'Inflation_lag{lag}' in temp_data.columns:
                    if j == 0:
                        temp_data[f'Inflation_lag{lag}'] = y.iloc[-lag]
                    else:
                        temp_data[f'Inflation_lag{lag}'] = bootstrap_predictions[j-1] if lag == 1 else temp_data[f'Inflation_lag{lag-1}']
            
            # Make prediction
            pred = model.predict(temp_data)[0]
            bootstrap_predictions.append(pred)
            bootstrap_forecasts[i, j] = pred
            
            # Update rolling features for next prediction
            for window in [3, 5, 7]:
                if f'rolling_mean_{window}' in temp_data.columns:
                    # Simplified update for demonstration
                    temp_data[f'rolling_mean_{window}'] = temp_data[f'Inflation_lag1']
                    temp_data[f'rolling_std_{window}'] = 0.1  # Small constant for stability
    
    # Calculate percentiles for confidence intervals
    lower_bound = np.percentile(bootstrap_forecasts, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_forecasts, 97.5, axis=0)
    median_forecast = np.median(bootstrap_forecasts, axis=0)
    
    # 4. Plot historical data and forecast
    plt.figure(figsize=(14, 7))
    
    # Historical data
    plt.plot(X['Year'], y, 'b-', label='Historical', linewidth=2)
    
    # Forecast
    future_years_list = list(future_years)
    plt.plot(future_years_list, median_forecast, 'r--', label='Forecast', linewidth=2)
    plt.fill_between(
        future_years_list, 
        lower_bound, 
        upper_bound, 
        color='red', 
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # Add markers for actual vs predicted on test set
    test_years = X_test['Year']
    plt.scatter(test_years, y_test, color='green', label='Test Data', zorder=5)
    plt.scatter(test_years, results[best_model_name]['y_pred'], 
               color='purple', label='Test Predictions', zorder=5)
    
    # Add error bars for test predictions
    for year, y_true, y_pred in zip(test_years, y_test, results[best_model_name]['y_pred']):
        plt.plot([year, year], [y_true, y_pred], 'k-', alpha=0.3)
    
    # Formatting
    plt.title(f'Inflation Forecast for {country} (Best Model: {best_model_name})', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text with model metrics
    metrics_text = (
        f"Model: {best_model_name}\n"
        f"RMSE: {results[best_model_name]['rmse']:.4f}\n"
        f"MAE: {results[best_model_name]['mae']:.4f}\n"
        f"R²: {results[best_model_name]['r2']:.4f}\n"
        f"MAPE: {results[best_model_name]['mape']:.2f}%"
    )
    plt.annotate(metrics_text, 
                xy=(0.02, 0.02), 
                xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('inflation_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Print forecast summary
    print("\n" + "="*50)
    print(f"{'Year':<10} {'Forecast':<15} {'Lower 95%':<15} {'Upper 95%':<15}")
    print("-"*50)
    for i, year in enumerate(future_years):
        print(f"{year:<10} {median_forecast[i]:<15.4f} {lower_bound[i]:<15.4f} {upper_bound[i]:<15.4f}")
    print("="*50)
    
    # 6. Save forecast to CSV
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Forecast': median_forecast,
        'Lower_95': lower_bound,
        'Upper_95': upper_bound
    })
    forecast_df.to_csv('inflation_forecast.csv', index=False)
    print("\nSaved forecast to 'inflation_forecast.csv'")
    
    # 7. Additional diagnostic plots
    try:
        # Rolling RMSE
        plt.figure(figsize=(14, 5))
        
        # Calculate rolling RMSE on test set
        test_rmse = np.sqrt((y_test - results[best_model_name]['y_pred'])**2)
        rolling_rmse = test_rmse.rolling(window=3).mean()
        
        plt.plot(test_years, test_rmse, 'o-', label='Test RMSE')
        plt.plot(test_years, rolling_rmse, 'r-', label='Rolling RMSE (window=3)')
        plt.axhline(y=results[best_model_name]['rmse'], color='g', linestyle='--', 
                   label=f'Overall RMSE: {results[best_model_name]["rmse"]:.4f}')
        
        plt.title('Model Performance Over Time', fontsize=14)
        plt.xlabel('Year')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('rolling_rmse.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in diagnostic plots: {str(e)}")
        
    print("\nForecast and analysis complete. Check the generated plots and CSV files for results.")
    print("Generated files:")
    print("- model_comparison.png: Model performance comparison")
    print("- feature_importance.png: Feature importance plots")
    print("- correlation_heatmap.png: Feature correlation matrix")
    print("- time_series_decomposition.png: Decomposition of time series")
    print("- residual_analysis.png: Analysis of model residuals")
    print("- inflation_forecast.png: Final forecast with confidence intervals")
    print("- rolling_rmse.png: Model performance over time")
    print("- model_results.csv: Detailed model performance metrics")
    print("- inflation_forecast.csv: Forecast values with confidence intervals")