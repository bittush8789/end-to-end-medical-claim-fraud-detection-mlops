import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureEngineering")

class FeatureEngineer:
    def __init__(self):
        pass

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features to the claims dataframe.
        Works on raw (not yet encoded/scaled) inputs.
        """
        logger.info("Adding engineered features...")
        df_feats = df.copy()

        # 1. Claim-to-Coverage Ratio
        # Ratios show if a claim is disproportionately high compared to policy coverage
        df_feats['claim_to_coverage_ratio'] = df_feats['claim_amount'] / (df_feats['coverage_amount'] + 1.0)

        # 2. Fraud History Score
        # Ratio of past fraud cases to total claims for a provider
        df_feats['fraud_history_score'] = df_feats['previous_fraud_cases'] / (df_feats['provider_claim_count'] + 1.0)

        # 3. Claim Frequency Score
        # High number of claims for younger patients is often suspicious (rapid claims activity)
        df_feats['claim_frequency_score'] = df_feats['previous_claims'] / (df_feats['patient_age'] + 1.0)

        # 4. Risk Score (Composite score)
        # Combines provider history, age risk, and claim size
        # We normalize values or use rank/scaled interactions to create an additive risk index
        df_feats['risk_score'] = (
            (df_feats['previous_fraud_cases'] * 5.0) +
            (df_feats['emergency_admission'] * 2.0) +
            (df_feats['claim_to_coverage_ratio'] * 10.0)
        )

        # 5. Average Claim Ratio
        # Let's calculate the mean claim amount by procedure code
        # During prediction, we can use predefined mapping.
        # Here we compute it dynamically or fallback to overall average if procedure_code not in map
        proc_mean_map = {
            'P001': 5000.0,
            'P002': 12000.0,
            'P003': 20000.0,
            'P004': 35000.0,
            'P005': 8000.0
        }
        df_feats['avg_claim_for_proc'] = df_feats['procedure_code'].map(proc_mean_map).fillna(15000.0)
        df_feats['average_claim_ratio'] = df_feats['claim_amount'] / (df_feats['avg_claim_for_proc'] + 1.0)
        
        # Drop temporary column
        df_feats = df_feats.drop(columns=['avg_claim_for_proc'])

        logger.info("Feature engineering complete.")
        return df_feats

if __name__ == "__main__":
    import os
    from data_ingestion import DataIngestion
    
    raw_path = "data/raw/claims_data.csv"
    if not os.path.exists(raw_path):
        logger.info("Raw data not found. Creating synthetic raw data...")
        ingestor = DataIngestion()
        df = ingestor.generate_synthetic_data()
    else:
        df = pd.read_csv(raw_path)
        
    fe = FeatureEngineer()
    df_feat = fe.add_features(df)
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    df_feat.to_csv(os.path.join(processed_dir, "engineered_claims_data.csv"), index=False)
    logger.info(f"Engineered dataset saved with columns: {list(df_feat.columns)}")
