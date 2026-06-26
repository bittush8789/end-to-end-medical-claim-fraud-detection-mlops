import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Defensive import for CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost is not available. Proceeding without CatBoost.")

from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from data_ingestion import DataIngestion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelTraining")

class ModelTrainer:
    def __init__(self, processed_data_path="data/processed/engineered_claims_data.csv", model_dir="models"):
        self.processed_data_path = processed_data_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.fe = FeatureEngineer()
        self.preprocessor = DataPreprocessor(model_dir=self.model_dir)

    def load_or_create_data(self):
        """
        Loads engineered data or runs ingestion and engineering pipelines.
        """
        if not os.path.exists(self.processed_data_path):
            logger.info("Engineered dataset not found. Re-running data pipeline...")
            raw_path = "data/raw/claims_data.csv"
            if not os.path.exists(raw_path):
                ingestor = DataIngestion()
                df = ingestor.generate_synthetic_data()
            else:
                df = pd.read_csv(raw_path)
            
            df_feat = self.fe.add_features(df)
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            df_feat.to_csv(self.processed_data_path, index=False)
        
        return pd.read_csv(self.processed_data_path)

    def train_and_select_best_model(self):
        df = self.load_or_create_data()
        
        # Split target and features
        X = df.drop(columns=['claim_id', 'is_fraud'])
        y = df['is_fraud']
        
        # Identify columns
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Fit preprocessor
        self.preprocessor.fit(X_train, num_cols, cat_cols)
        self.preprocessor.save()
        
        # Transform
        X_train_proc = self.preprocessor.transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        
        # Define candidate models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_seed=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=42)

        best_model_name = None
        best_model = None
        best_f1 = -1
        results = []

        logger.info("Starting model training and evaluation...")
        for name, clf in models.items():
            logger.info(f"Training {name}...")
            clf.fit(X_train_proc, y_train)
            
            # Predict
            preds = clf.predict(X_test_proc)
            probs = clf.predict_proba(X_test_proc)[:, 1]
            
            # Scores
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, probs)
            
            logger.info(f"{name} Metrics -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
            
            results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC AUC': roc_auc
            })
            
            # Select best model based on F1-Score
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_model_name = name
        
        logger.info(f"Best Model Selected: {best_model_name} with F1-Score: {best_f1:.4f}")
        
        # Save best model
        best_model_path = os.path.join(self.model_dir, "trained_model.pkl")
        joblib.dump(best_model, best_model_path)
        logger.info(f"Saved best model ({best_model_name}) to {best_model_path}")
        
        # Save model details
        details_path = os.path.join(self.model_dir, "model_details.pkl")
        joblib.dump({
            'model_name': best_model_name,
            'features': list(X_train_proc.columns),
            'num_cols': num_cols,
            'cat_cols': cat_cols
        }, details_path)
        
        # Convert results to DataFrame and save summary
        results_df = pd.DataFrame(results)
        os.makedirs("reports", exist_ok=True)
        results_df.to_csv("reports/model_comparison.csv", index=False)
        
        return best_model, results_df

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_select_best_model()
