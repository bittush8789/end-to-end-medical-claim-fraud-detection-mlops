import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictionService")

class FraudPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.fe = FeatureEngineer()
        
        # Load artifacts
        logger.info("Loading ML pipeline artifacts...")
        self.preprocessor = DataPreprocessor(model_dir=self.model_dir).load()
        self.model = joblib.load(os.path.join(self.model_dir, "trained_model.pkl"))
        self.details = joblib.load(os.path.join(self.model_dir, "model_details.pkl"))
        logger.info("ML pipeline artifacts loaded successfully.")

    def predict(self, claim_data: dict) -> dict:
        """
        Predicts fraud for a single claim dictionary.
        Input format example:
        {
            'patient_age': 45,
            'gender': 'Male',
            'policy_type': 'Gold',
            'policy_tenure': 24,
            'previous_claims': 2,
            'claim_amount': 15000.0,
            'hospital_type': 'Private',
            'procedure_code': 'P002',
            'diagnosis_code': 'D002',
            'length_of_stay': 5,
            'emergency_admission': 1,
            'provider_id': 'PROV0010',
            'provider_claim_count': 50,
            'previous_fraud_cases': 1,
            'coverage_amount': 50000.0,
            'deductible_amount': 500
        }
        """
        # Convert dictionary to DataFrame
        df_raw = pd.DataFrame([claim_data])
        
        # Feature Engineering
        df_feat = self.fe.add_features(df_raw)
        
        # Preprocessing
        df_proc = self.preprocessor.transform(df_feat)
        
        # Predict Class and Probability
        prediction = int(self.model.predict(df_proc)[0])
        probability = float(self.model.predict_proba(df_proc)[0][1])
        
        # Risk level determination
        if probability < 0.35:
            risk_level = "Low Risk"
        elif probability < 0.70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        # Explanations (Fraud Indicators)
        explanations = []
        if claim_data['claim_amount'] > claim_data['coverage_amount']:
            explanations.append("Claim Amount exceeds Coverage limit.")
        elif claim_data['claim_amount'] / (claim_data['coverage_amount'] + 1.0) > 0.8:
            explanations.append("High Claim-to-Coverage Ratio.")
            
        if claim_data['previous_fraud_cases'] >= 5:
            explanations.append("Provider has high history of previous fraud cases.")
        elif claim_data['previous_fraud_cases'] >= 2:
            explanations.append("Provider has prior fraud cases on record.")
            
        if claim_data['previous_claims'] > 6:
            explanations.append("Patient has high frequency of previous claims.")
            
        if claim_data['procedure_code'] == 'P004' and claim_data['diagnosis_code'] == 'D001':
            explanations.append("Unusual procedure pattern: Expensive procedure paired with a minor diagnosis code.")
            
        if claim_data['length_of_stay'] > 12 and claim_data['hospital_type'] == 'Private':
            explanations.append("Extended length of stay at a Private facility.")

        # Default explanation if probability is elevated but no specific heuristic was hit
        if probability > 0.40 and not explanations:
            explanations.append("Elevated risk due to general multivariate clinical and demographic variables.")
        elif not explanations:
            explanations.append("No critical risk factors identified. Claim patterns are standard.")

        result = {
            'is_fraud': prediction,
            'fraud_probability': round(probability * 100, 1),
            'risk_level': risk_level,
            'explanations': explanations,
            'model_used': self.details.get('model_name')
        }
        
        return result

if __name__ == "__main__":
    # Quick test case
    test_claim = {
        'patient_age': 25,
        'gender': 'Male',
        'policy_type': 'Bronze',
        'policy_tenure': 6,
        'previous_claims': 8,
        'claim_amount': 25000.0,
        'hospital_type': 'Private',
        'procedure_code': 'P004',
        'diagnosis_code': 'D001',
        'length_of_stay': 15,
        'emergency_admission': 1,
        'provider_id': 'PROV0010',
        'provider_claim_count': 120,
        'previous_fraud_cases': 8,
        'coverage_amount': 20000.0,
        'deductible_amount': 1000
    }
    
    # Check if models exist
    if os.path.exists("models/trained_model.pkl"):
        predictor = FraudPredictor()
        res = predictor.predict(test_claim)
        print("Prediction result:", res)
    else:
        print("Model has not been trained yet. Run train.py first.")
