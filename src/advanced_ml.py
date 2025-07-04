"""
Advanced Model Training with Enhanced Features
Includes ensemble methods, hyperparameter tuning, and advanced evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import VotingClassifier, VotingRegressor, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import optuna
from typing import Dict, Tuple, List
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """
    Advanced model training with hyperparameter optimization and ensemble methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_models = {}
        self.ensemble_models = {}
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced feature engineering
        """
        print("🔧 Creating advanced features...")
        
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).copy()
        
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_data = df[symbol_mask].copy()
            
            # Price momentum features
            for window in [3, 7, 14]:
                df.loc[symbol_mask, f'price_momentum_{window}d'] = (
                    symbol_data['close'].pct_change(window)
                )
            
            # Volatility regime features
            df.loc[symbol_mask, 'volatility_regime'] = np.where(
                symbol_data['volatility_20d'] > symbol_data['volatility_20d'].rolling(50).mean(),
                1, 0
            )
            
            # Volume-price confirmation
            df.loc[symbol_mask, 'volume_price_confirmation'] = np.where(
                (symbol_data['volume'] > symbol_data['volume'].rolling(20).mean()) &
                (symbol_data['close'] > symbol_data['close'].shift(1)),
                1, -1
            )
            
            # Sentiment momentum
            if 'overall_sentiment' in df.columns:
                df.loc[symbol_mask, 'sentiment_momentum_3d'] = (
                    symbol_data['overall_sentiment'].rolling(3).mean() - 
                    symbol_data['overall_sentiment'].rolling(7).mean()
                )
        
        print(f"✅ Advanced features created. New shape: {df.shape}")
        return df
    
    def optimize_hyperparameters_optuna(self, 
                                       X_train: np.ndarray, 
                                       y_train: np.ndarray,
                                       model_type: str = 'classification') -> Dict:
        """
        Use Optuna for hyperparameter optimization
        """
        def objective(trial):
            if model_type == 'classification':
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    random_state=self.random_state
                )
            else:
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    random_state=self.random_state
                )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                
                if model_type == 'classification':
                    score = model.score(X_val, y_val)
                else:
                    from sklearn.metrics import r2_score
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        return study.best_params
    
    def create_ensemble_models(self, 
                              X_train: np.ndarray, 
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              model_type: str = 'classification') -> Dict:
        """
        Create ensemble models for better performance
        """
        print(f"🎯 Creating ensemble {model_type} models...")
        
        if model_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Base models
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            xgb = XGBClassifier(n_estimators=100, random_state=self.random_state)
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Voting classifier
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
                voting='soft'
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            train_score = ensemble.score(X_train, y_train)
            test_score = ensemble.score(X_test, y_test)
            
            return {
                'model': ensemble,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'predictions': ensemble.predict(X_test),
                'probabilities': ensemble.predict_proba(X_test)
            }
            
        else:  # regression
            from sklearn.ensemble import RandomForestRegressor
            from xgboost import XGBRegressor
            
            # Base models
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            xgb = XGBRegressor(n_estimators=100, random_state=self.random_state)
            en = ElasticNet(random_state=self.random_state)
            
            # Voting regressor
            ensemble = VotingRegressor(
                estimators=[('rf', rf), ('xgb', xgb), ('en', en)]
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            from sklearn.metrics import r2_score
            train_pred = ensemble.predict(X_train)
            test_pred = ensemble.predict(X_test)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            return {
                'model': ensemble,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'predictions': test_pred
            }
    
    def train_advanced_models(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             feature_names: List[str]) -> Dict:
        """
        Train advanced models with optimization
        """
        results = {}
        
        print("🚀 Starting Advanced Model Training...")
        
        # 1. Hyperparameter optimization for classification
        print("\n🔍 Optimizing hyperparameters for classification...")
        try:
            best_params_cls = self.optimize_hyperparameters_optuna(
                X_train, y_train, 'classification'
            )
            print(f"✅ Best classification params: {best_params_cls}")
        except Exception as e:
            print(f"⚠️ Hyperparameter optimization failed: {e}")
            best_params_cls = {}
        
        # 2. Train optimized XGBoost
        from xgboost import XGBClassifier
        optimized_xgb = XGBClassifier(**best_params_cls, random_state=self.random_state)
        optimized_xgb.fit(X_train, y_train)
        
        results['optimized_xgb'] = {
            'model': optimized_xgb,
            'train_accuracy': optimized_xgb.score(X_train, y_train),
            'test_accuracy': optimized_xgb.score(X_test, y_test),
            'feature_importance': pd.DataFrame({
                'feature': feature_names,
                'importance': optimized_xgb.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        # 3. Ensemble models
        ensemble_results = self.create_ensemble_models(
            X_train, y_train, X_test, y_test, 'classification'
        )
        results['ensemble'] = ensemble_results
        
        # 4. Advanced evaluation
        results['evaluation'] = self._advanced_evaluation(
            y_test, results['ensemble']['predictions'], 
            results['ensemble']['probabilities']
        )
        
        print("✅ Advanced model training completed!")
        return results
    
    def _advanced_evaluation(self, y_true, y_pred, y_proba) -> Dict:
        """
        Advanced model evaluation metrics
        """
        from sklearn.metrics import (
            precision_recall_curve, roc_auc_score, 
            confusion_matrix, f1_score
        )
        
        evaluation = {
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_proba[:, 1]) if y_proba.shape[1] > 1 else None
        }
        
        return evaluation
    
    def save_advanced_models(self, results: Dict, model_dir: str = "models"):
        """
        Save advanced models and results
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble model
        if 'ensemble' in results:
            joblib.dump(results['ensemble']['model'], 
                       f"{model_dir}/ensemble_model.joblib")
        
        # Save optimized XGBoost
        if 'optimized_xgb' in results:
            joblib.dump(results['optimized_xgb']['model'], 
                       f"{model_dir}/optimized_xgb_model.joblib")
        
        # Save evaluation results
        import json
        evaluation_data = {
            'ensemble_accuracy': results['ensemble']['test_accuracy'],
            'optimized_xgb_accuracy': results['optimized_xgb']['test_accuracy'],
            'f1_score': results['evaluation']['f1_score'],
            'roc_auc': results['evaluation']['roc_auc']
        }
        
        with open(f"{model_dir}/advanced_model_results.json", 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"✅ Advanced models saved to {model_dir}/")

# Example usage
if __name__ == "__main__":
    # This would be integrated into the main notebook
    print("🔬 Advanced Model Training Module Ready!")
    print("Import this module in your notebook for enhanced ML capabilities.")
