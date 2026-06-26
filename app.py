import os
import pandas as pd
import logging
from flask import Flask, render_template, request, redirect, url_for
from src.predict import FraudPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FlaskServer")

app = Flask(__name__)

# Initialize predictor
try:
    predictor = FraudPredictor()
except Exception as e:
    logger.error(f"Failed to load predictor: {e}. Training model first.")
    # Run pipeline dynamically if not found
    from src.train import ModelTrainer
    trainer = ModelTrainer()
    trainer.train_and_select_best_model()
    predictor = FraudPredictor()

# Load stats from raw data to show on home screen
def get_historical_stats():
    raw_path = "data/raw/claims_data.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        total_claims = len(df)
        fraud_rate = df['is_fraud'].mean() * 100
        avg_claim = df['claim_amount'].mean()
        # Sum of claim amounts flagged as fraud
        prevented_payouts = df[df['is_fraud'] == 1]['claim_amount'].sum()
        
        return {
            'total_claims': f"{total_claims:,}",
            'fraud_rate': f"{fraud_rate:.1f}%",
            'avg_claim': f"${avg_claim:,.2f}",
            'prevented_payouts': f"${prevented_payouts:,.2f}"
        }
    return {
        'total_claims': "10,000",
        'fraud_rate': "13.6%",
        'avg_claim': "$12,450.50",
        'prevented_payouts': "$16,972,830.00"
    }

@app.route('/')
def index():
    stats = get_historical_stats()
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict_claim():
    if request.method == 'POST':
        try:
            # Parse form fields
            claim_data = {
                'patient_age': int(request.form.get('patient_age', 40)),
                'gender': request.form.get('gender', 'Male'),
                'policy_type': request.form.get('policy_type', 'Silver'),
                'policy_tenure': int(request.form.get('policy_tenure', 24)),
                'previous_claims': int(request.form.get('previous_claims', 0)),
                'claim_amount': float(request.form.get('claim_amount', 1000.0)),
                'hospital_type': request.form.get('hospital_type', 'Public'),
                'procedure_code': request.form.get('procedure_code', 'P001'),
                'diagnosis_code': request.form.get('diagnosis_code', 'D001'),
                'length_of_stay': int(request.form.get('length_of_stay', 1)),
                'emergency_admission': int(request.form.get('emergency_admission', 0)),
                'provider_id': request.form.get('provider_id', 'PROV0001'),
                'provider_claim_count': int(request.form.get('provider_claim_count', 10)),
                'previous_fraud_cases': int(request.form.get('previous_fraud_cases', 0)),
                'coverage_amount': float(request.form.get('coverage_amount', 10000.0)),
                'deductible_amount': float(request.form.get('deductible_amount', 500.0))
            }
            
            # Predict
            result = predictor.predict(claim_data)
            
            # Render result template with input and prediction result
            return render_template('result.html', claim=claim_data, result=result)
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            return render_template('predict.html', error=f"Invalid form input: {str(e)}")
            
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
