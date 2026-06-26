import os
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib
matplotlib.use('Agg') # Safe for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)

from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelEvaluation")

class ModelEvaluator:
    def __init__(self, processed_data_path="data/processed/engineered_claims_data.csv", model_dir="models", reports_dir="reports"):
        self.processed_data_path = processed_data_path
        self.model_dir = model_dir
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(self.reports_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Load preprocessor, model, and details
        self.preprocessor = DataPreprocessor(model_dir=self.model_dir).load()
        self.model = joblib.load(os.path.join(self.model_dir, "trained_model.pkl"))
        self.details = joblib.load(os.path.join(self.model_dir, "model_details.pkl"))

    def evaluate(self):
        logger.info("Loading dataset and splitting into test set...")
        df = pd.read_csv(self.processed_data_path)
        
        # Split target and features
        X = df.drop(columns=['claim_id', 'is_fraud'])
        y = df['is_fraud']
        
        # Use exact same split logic to get the same test partition
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Transform test set
        X_test_proc = self.preprocessor.transform(X_test)
        
        logger.info("Generating predictions and scoring...")
        preds = self.model.predict(X_test_proc)
        probs = self.model.predict_proba(X_test_proc)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        
        # Create visual figures
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve(y_test, probs)
        self.plot_feature_importances(X_test_proc.columns)
        
        # Save evaluation report text
        report_path = os.path.join(self.reports_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write("==================================================\n")
            f.write("      MEDICAL CLAIM FRAUD DETECTION REPORT        \n")
            f.write("==================================================\n\n")
            f.write(f"Best Model Selected: {self.details.get('model_name')}\n\n")
            f.write("--- Overall Performance Metrics (Test Set) ---\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"ROC AUC:   {roc_auc:.4f}\n\n")
            f.write("--- Classification Report ---\n")
            f.write(classification_report(y_test, preds))
            f.write("\n--- Confusion Matrix ---\n")
            f.write(f"True Negatives (Legit): {cm[0][0]}\n")
            f.write(f"False Positives:        {cm[0][1]}\n")
            f.write(f"False Negatives:        {cm[1][0]}\n")
            f.write(f"True Positives (Fraud):  {cm[1][1]}\n")
            
        logger.info(f"Saved evaluation report text to {report_path}")
        
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraudulent'], 
                    yticklabels=['Legitimate', 'Fraudulent'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {path}")
        
    def plot_roc_curve(self, y_test, probs):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "roc_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved ROC curve plot to {path}")
        
    def plot_feature_importances(self, feature_names):
        # Only draw if model has feature_importances_ or coef_
        importances = None
        title = "Feature Importance"
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            title = "Feature Coefficients (Magnitude)"
            
        if importances is not None:
            # Map features to names and sort
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
            plt.title(f'Top 15 Feature Importances ({self.details.get("model_name")})')
            plt.xlabel('Relative Importance / Coefficient Magnitude')
            plt.ylabel('Features')
            plt.tight_layout()
            path = os.path.join(self.figures_dir, "feature_importance.png")
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info(f"Saved feature importance plot to {path}")
        else:
            logger.info("Model does not support feature importances plotting.")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()
