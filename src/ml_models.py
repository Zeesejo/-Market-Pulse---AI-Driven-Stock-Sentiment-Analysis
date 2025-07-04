"""
Market Pulse - Machine Learning Module

This module implements various machine learning models to predict stock price
movements based on sentiment analysis and technical indicators.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, mean_squared_error, 
                           mean_absolute_error, r2_score, classification_report,
                           confusion_matrix)
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates features for machine learning models"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
    
    def create_price_targets(self, data: pd.DataFrame, prediction_days: int = 1) -> pd.DataFrame:
        """
        Create price prediction targets
        
        Args:
            data: DataFrame with stock price data
            prediction_days: Number of days ahead to predict
        
        Returns:
            DataFrame with target variables
        """
        df = data.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        # Price direction (classification target)
        df['future_close'] = df.groupby('Symbol')['Close'].shift(-prediction_days)
        df['price_change'] = (df['future_close'] - df['Close']) / df['Close']
        
        # Classification targets
        df['price_direction'] = (df['price_change'] > 0).astype(int)  # 1 if up, 0 if down
        
        # Multi-class classification
        def categorize_movement(change):
            if change > 0.02:  # > 2% increase
                return 2  # Strong Up
            elif change > 0:
                return 1  # Weak Up
            elif change > -0.02:
                return 0  # Weak Down
            else:
                return -1  # Strong Down
        
        df['price_movement_category'] = df['price_change'].apply(categorize_movement)
        
        # Regression targets
        df['future_return'] = df['price_change']
        df['future_volatility'] = df.groupby('Symbol')['price_change'].rolling(window=5).std().reset_index(0, drop=True)
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series
        
        Args:
            data: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        df = data.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('Symbol')[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, columns: List[str], 
                              windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            data: DataFrame with time series data
            columns: Columns to create rolling features for
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling features
        """
        df = data.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df.groupby('Symbol')[col].rolling(window=window).mean().reset_index(0, drop=True)
                df[f'{col}_rolling_std_{window}'] = df.groupby('Symbol')[col].rolling(window=window).std().reset_index(0, drop=True)
                df[f'{col}_rolling_min_{window}'] = df.groupby('Symbol')[col].rolling(window=window).min().reset_index(0, drop=True)
                df[f'{col}_rolling_max_{window}'] = df.groupby('Symbol')[col].rolling(window=window).max().reset_index(0, drop=True)
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between sentiment and technical indicators
        
        Args:
            data: DataFrame with features
        
        Returns:
            DataFrame with interaction features
        """
        df = data.copy()
        
        # Sentiment-Technical interactions
        if 'daily_sentiment' in df.columns and 'RSI' in df.columns:
            df['sentiment_rsi_interaction'] = df['daily_sentiment'] * df['RSI']
        
        if 'daily_sentiment' in df.columns and 'MACD' in df.columns:
            df['sentiment_macd_interaction'] = df['daily_sentiment'] * df['MACD']
        
        if 'news_volume' in df.columns and 'Volume' in df.columns:
            df['news_volume_interaction'] = df['news_volume'] * df['Volume']
        
        # Price momentum features
        if 'Close' in df.columns:
            df['price_momentum_5d'] = df.groupby('Symbol')['Close'].pct_change(periods=5)
            df['price_momentum_10d'] = df.groupby('Symbol')['Close'].pct_change(periods=10)
        
        return df
    
    def prepare_features(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare all features for machine learning
        
        Args:
            stock_data: DataFrame with stock price and technical indicators
            sentiment_data: DataFrame with sentiment analysis results
        
        Returns:
            DataFrame with all features ready for ML
        """
        logger.info("Preparing features for machine learning...")
        
        # Start with stock data
        df = stock_data.copy()
        
        # Merge with sentiment data if available
        if sentiment_data is not None and not sentiment_data.empty:
            # Convert date column to match
            sentiment_data = sentiment_data.copy()
            sentiment_data['Date'] = pd.to_datetime(sentiment_data['date'])
            
            # Merge on symbol and date
            df = pd.merge(df, sentiment_data, on=['Symbol', 'Date'], how='left')
            
            # Fill missing sentiment values with neutral
            sentiment_columns = [col for col in sentiment_data.columns if 'sentiment' in col.lower()]
            for col in sentiment_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        
        # Create price targets
        df = self.create_price_targets(df)
        
        # Create lag features for important indicators
        lag_columns = ['Close', 'Volume', 'RSI', 'MACD']
        if 'daily_sentiment' in df.columns:
            lag_columns.append('daily_sentiment')
        
        available_lag_columns = [col for col in lag_columns if col in df.columns]
        df = self.create_lag_features(df, available_lag_columns, [1, 2, 3, 5])
        
        # Create rolling features
        rolling_columns = ['daily_sentiment', 'news_volume'] if 'daily_sentiment' in df.columns else []
        rolling_columns.extend(['Close', 'Volume'])
        available_rolling_columns = [col for col in rolling_columns if col in df.columns]
        df = self.create_rolling_features(df, available_rolling_columns, [3, 5, 10])
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Add time-based features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Encode categorical variables
        if 'Symbol' in df.columns:
            if 'symbol_encoder' not in self.label_encoders:
                self.label_encoders['symbol_encoder'] = LabelEncoder()
                df['symbol_encoded'] = self.label_encoders['symbol_encoder'].fit_transform(df['Symbol'])
            else:
                df['symbol_encoded'] = self.label_encoders['symbol_encoder'].transform(df['Symbol'])
        
        logger.info(f"Feature preparation completed. Shape: {df.shape}")
        return df

class ModelTrainer:
    """Trains and evaluates machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_columns = ['price_direction', 'price_movement_category', 'future_return']
    
    def select_features(self, data: pd.DataFrame) -> List[str]:
        """
        Select relevant features for modeling
        
        Args:
            data: DataFrame with all features
        
        Returns:
            List of selected feature names
        """
        # Technical indicators
        technical_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Price_Change', 'Price_Change_1d', 'Price_Change_5d',
            'Volume_Ratio'
        ]
        
        # Sentiment features
        sentiment_features = [
            'daily_sentiment', 'sentiment_volatility', 'news_volume',
            'avg_sentiment_strength', 'composite_positive', 'composite_negative',
            'vader_compound', 'textblob_polarity'
        ]
        
        # Lag features
        lag_features = [col for col in data.columns if '_lag_' in col]
        
        # Rolling features
        rolling_features = [col for col in data.columns if '_rolling_' in col]
        
        # Interaction features
        interaction_features = [col for col in data.columns if '_interaction' in col]
        
        # Time features
        time_features = ['day_of_week', 'month', 'quarter', 'symbol_encoded']
        
        # Momentum features
        momentum_features = [col for col in data.columns if 'momentum' in col]
        
        # Combine all feature categories
        all_features = (technical_features + sentiment_features + lag_features + 
                       rolling_features + interaction_features + time_features + momentum_features)
        
        # Filter to only include features that exist in the data
        selected_features = [feat for feat in all_features if feat in data.columns]
        
        logger.info(f"Selected {len(selected_features)} features for modeling")
        return selected_features
    
    def prepare_data_for_training(self, data: pd.DataFrame, target_col: str, 
                                test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing
        
        Args:
            data: DataFrame with features and targets
            target_col: Target column name
            test_size: Proportion of data for testing
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove rows with missing targets
        clean_data = data.dropna(subset=[target_col])
        
        # Select features
        feature_columns = self.select_features(clean_data)
        self.feature_names = feature_columns
        
        # Prepare features and target
        X = clean_data[feature_columns].fillna(0)  # Fill remaining NaN with 0
        y = clean_data[target_col]
        
        # Time series split (use last 20% as test set)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler_name = f"scaler_{target_col}"
        if scaler_name not in self.scalers:
            self.scalers[scaler_name] = RobustScaler()
            X_train_scaled = self.scalers[scaler_name].fit_transform(X_train)
        else:
            X_train_scaled = self.scalers[scaler_name].transform(X_train)
        
        X_test_scaled = self.scalers[scaler_name].transform(X_test)
        
        logger.info(f"Data prepared for {target_col}: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def train_classification_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray,
                                  target_name: str) -> Dict[str, Dict]:
        """
        Train classification models for price direction prediction
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            target_name: Name of the target variable
        
        Returns:
            Dictionary with model results
        """
        logger.info(f"Training classification models for {target_name}...")
        
        results = {}
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                                 class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced',
                                                   max_iter=1000),
        }
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics
                }
                
                # Store best model
                model_key = f"{target_name}_{model_name}"
                self.models[model_key] = model
                
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def train_regression_models(self, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              target_name: str) -> Dict[str, Dict]:
        """
        Train regression models for price return prediction
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            target_name: Name of the target variable
        
        Returns:
            Dictionary with model results
        """
        logger.info(f"Training regression models for {target_name}...")
        
        results = {}
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
        }
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'metrics': metrics
                }
                
                # Store model
                model_key = f"{target_name}_{model_name}"
                self.models[model_key] = model
                
                logger.info(f"{model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model_name: Name of the trained model
        
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_models(self, save_dir: str = "models"):
        """
        Save trained models and scalers
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f"{scaler_name}.joblib")
            joblib.dump(scaler, scaler_path)
        
        # Save feature names
        feature_path = os.path.join(save_dir, "feature_names.joblib")
        joblib.dump(self.feature_names, feature_path)
        
        logger.info(f"Models saved to {save_dir}")

def main():
    """Main function to demonstrate machine learning pipeline"""
    print("🤖 Market Pulse - Machine Learning Demo")
    print("="*50)
    
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Sample stock data
    dates = pd.date_range(start='2023-01-01', end='2024-07-01', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    sample_data = []
    for symbol in symbols:
        for date in dates:
            if date.weekday() < 5:  # Only weekdays
                price = 100 + np.random.normal(0, 5)
                sample_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Open': price,
                    'High': price + np.random.uniform(0, 2),
                    'Low': price - np.random.uniform(0, 2),
                    'Close': price + np.random.normal(0, 1),
                    'Volume': np.random.randint(1000000, 10000000),
                    'RSI': np.random.uniform(30, 70),
                    'MACD': np.random.normal(0, 0.5),
                    'SMA_20': price,
                    'EMA_12': price,
                    'BB_Upper': price + 2,
                    'BB_Lower': price - 2,
                    'Price_Change': np.random.normal(0, 0.02)
                })
    
    stock_df = pd.DataFrame(sample_data)
    
    # Sample sentiment data
    sentiment_data = []
    for symbol in symbols:
        for date in dates[::2]:  # Every other day
            if date.weekday() < 5:
                sentiment_data.append({
                    'date': date.date(),
                    'symbol': symbol,
                    'daily_sentiment': np.random.normal(0, 0.3),
                    'news_volume': np.random.randint(1, 20),
                    'sentiment_volatility': np.random.uniform(0, 0.5),
                    'composite_positive': np.random.uniform(0, 1),
                    'composite_negative': np.random.uniform(0, 1),
                    'vader_compound': np.random.normal(0, 0.4),
                    'textblob_polarity': np.random.normal(0, 0.3)
                })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Initialize feature engineer and model trainer
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    
    # Prepare features
    print("\n🔧 Engineering Features...")
    features_df = feature_engineer.prepare_features(stock_df, sentiment_df)
    
    print(f"✅ Feature engineering completed. Shape: {features_df.shape}")
    print(f"📊 Available features: {len(model_trainer.select_features(features_df))}")
    
    # Train classification models
    print("\n🎯 Training Classification Models (Price Direction)...")
    X_train, X_test, y_train, y_test = model_trainer.prepare_data_for_training(
        features_df, 'price_direction'
    )
    
    classification_results = model_trainer.train_classification_models(
        X_train, X_test, y_train, y_test, 'price_direction'
    )
    
    # Train regression models
    print("\n📈 Training Regression Models (Future Returns)...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = model_trainer.prepare_data_for_training(
        features_df, 'future_return'
    )
    
    regression_results = model_trainer.train_regression_models(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, 'future_return'
    )
    
    # Display results
    print("\n📊 Model Performance Summary:")
    print("\nClassification Results (Price Direction):")
    for model_name, results in classification_results.items():
        metrics = results['metrics']
        print(f"  {model_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\nRegression Results (Future Returns):")
    for model_name, results in regression_results.items():
        metrics = results['metrics']
        print(f"  {model_name}:")
        print(f"    R²: {metrics['r2']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    MAE: {metrics['mae']:.4f}")
    
    # Show feature importance
    if classification_results:
        best_model = list(classification_results.keys())[0]
        model_key = f"price_direction_{best_model}"
        feature_importance = model_trainer.get_feature_importance(model_key)
        
        if not feature_importance.empty:
            print(f"\n🔍 Top 10 Important Features ({best_model}):")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save models
    print("\n💾 Saving Models...")
    model_trainer.save_models()
    
    print("\n🎉 Machine learning pipeline completed!")
    print("Models are ready for prediction and deployment")

if __name__ == "__main__":
    main()
