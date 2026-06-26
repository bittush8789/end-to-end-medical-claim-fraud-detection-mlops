import os
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataIngestion")

class DataIngestion:
    def __init__(self, raw_data_path="data/raw/claims_data.csv"):
        self.raw_data_path = raw_data_path
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

    def generate_synthetic_data(self, num_samples=10000, random_seed=42):
        """
        Generates realistic synthetic medical insurance claims data with embedded fraud patterns.
        """
        logger.info(f"Generating {num_samples} synthetic medical claim records...")
        np.random.seed(random_seed)

        # Generate fields
        claim_ids = [f"CLM{str(i).zfill(6)}" for i in range(1, num_samples + 1)]
        patient_ages = np.random.randint(18, 85, size=num_samples)
        genders = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.48, 0.52])
        policy_types = np.random.choice(['Gold', 'Silver', 'Bronze'], size=num_samples, p=[0.3, 0.5, 0.2])
        policy_tenures = np.random.randint(1, 120, size=num_samples) # in months
        
        # Provider variables
        num_providers = 100
        providers = [f"PROV{str(i).zfill(4)}" for i in range(1, num_providers + 1)]
        # Assign some providers as "high risk" (higher history of fraud)
        provider_probs = np.random.dirichlet(np.ones(num_providers)) # probability of a claim coming from this provider
        provider_ids = np.random.choice(providers, size=num_samples, p=provider_probs)
        
        # Calculate provider claim count and previous fraud history mapping
        provider_counts = pd.Series(provider_ids).value_counts()
        provider_fraud_map = {}
        for p in providers:
            # high risk providers (10% of them) get high fraud cases
            if int(p[4:]) % 10 == 0:
                provider_fraud_map[p] = np.random.randint(3, 12)
            else:
                provider_fraud_map[p] = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        previous_fraud_cases = [provider_fraud_map[pid] for pid in provider_ids]
        provider_claim_counts = [provider_counts[pid] for pid in provider_ids]

        # Patient previous claims
        previous_claims = np.random.negative_binomial(n=2, p=0.4, size=num_samples) # skewed distribution
        
        # Financial variables
        coverage_amounts = np.random.randint(5000, 150000, size=num_samples)
        deductible_amounts = np.random.randint(100, 5000, size=num_samples)
        
        # Claim amounts (often correlated with coverage but with random variations)
        claim_amounts = np.zeros(num_samples)
        for i in range(num_samples):
            # Normally claim is a fraction of coverage, but sometimes exceeds it
            base_claim = np.random.exponential(scale=15000) + 500
            # Clip or cap to logical amounts, but allow anomalies
            if np.random.rand() < 0.05:
                # Anomaly: claim amount much larger than coverage
                claim_amounts[i] = coverage_amounts[i] * np.random.uniform(1.1, 2.5)
            else:
                claim_amounts[i] = min(base_claim, coverage_amounts[i] * np.random.uniform(0.1, 0.95))
        
        claim_amounts = np.round(claim_amounts, 2)
        
        # Clinical variables
        hospital_types = np.random.choice(['Public', 'Private', 'Specialty'], size=num_samples, p=[0.4, 0.45, 0.15])
        procedure_codes = np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], size=num_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        diagnosis_codes = np.random.choice(['D001', 'D002', 'D003', 'D004', 'D005'], size=num_samples, p=[0.35, 0.25, 0.2, 0.12, 0.08])
        length_of_stays = np.random.randint(1, 21, size=num_samples)
        # Adjust length of stay for emergency
        emergency_admissions = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
        
        # Injecting Fraud Probability Logic
        is_fraud = np.zeros(num_samples, dtype=int)
        
        for i in range(num_samples):
            pid = provider_ids[i]
            p_fraud_val = provider_fraud_map[pid]
            
            # Base probability of fraud
            p_fraud = 0.04
            
            # 1. High provider fraud history
            if p_fraud_val >= 5:
                p_fraud += 0.35
            elif p_fraud_val >= 2:
                p_fraud += 0.15
                
            # 2. Claim exceeds coverage amount
            if claim_amounts[i] > coverage_amounts[i]:
                p_fraud += 0.30
            elif claim_amounts[i] / coverage_amounts[i] > 0.85:
                p_fraud += 0.15
                
            # 3. Medical anomalies (P004 is expensive, paired with D001 which is minor)
            if procedure_codes[i] == 'P004' and diagnosis_codes[i] == 'D001':
                p_fraud += 0.25
            
            # 4. Rapid repetitive claims (high previous claims for young patient)
            if previous_claims[i] > 6 and patient_ages[i] < 30:
                p_fraud += 0.20
                
            # 5. Over-stay in private hospital for minor diagnosis
            if length_of_stays[i] > 12 and hospital_types[i] == 'Private' and diagnosis_codes[i] in ['D001', 'D002']:
                p_fraud += 0.18

            # Cap probability between 0.01 and 0.95
            p_fraud = max(min(p_fraud, 0.95), 0.01)
            
            # Simulate fraud label
            is_fraud[i] = np.random.choice([0, 1], p=[1 - p_fraud, p_fraud])

        # Create DataFrame
        df = pd.DataFrame({
            'claim_id': claim_ids,
            'patient_age': patient_ages,
            'gender': genders,
            'policy_type': policy_types,
            'policy_tenure': policy_tenures,
            'claim_amount': claim_amounts,
            'hospital_type': hospital_types,
            'procedure_code': procedure_codes,
            'diagnosis_code': diagnosis_codes,
            'length_of_stay': length_of_stays,
            'emergency_admission': emergency_admissions,
            'provider_id': provider_ids,
            'provider_claim_count': provider_claim_counts,
            'previous_fraud_cases': previous_fraud_cases,
            'previous_claims': previous_claims,
            'coverage_amount': coverage_amounts,
            'deductible_amount': deductible_amounts,
            'is_fraud': is_fraud
        })
        
        # Save to csv
        df.to_csv(self.raw_data_path, index=False)
        logger.info(f"Synthetic dataset saved successfully to {self.raw_data_path}")
        logger.info(f"Fraud distribution:\n{df['is_fraud'].value_counts(normalize=True)}")
        return df

if __name__ == "__main__":
    ingestor = DataIngestion()
    ingestor.generate_synthetic_data()
