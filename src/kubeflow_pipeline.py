import kfp
from kfp import dsl
from kfp.dsl import component, Output, Artifact, Dataset, Model, Metrics

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def ingest_data_step(raw_data_path: Output[Dataset], train_data_path: Output[Dataset], test_data_path: Output[Dataset]):
    """Ingest synthetic dataset and split it."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Simulate Ingestion logic from src/data_ingestion.py
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'patient_age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'policy_type': np.random.choice(['Basic', 'Premium', 'Gold'], n_samples),
        'policy_tenure': np.random.uniform(0.5, 20.0, n_samples),
        'previous_claims': np.random.randint(0, 10, n_samples),
        'claim_amount': np.random.uniform(500, 50000, n_samples),
        'coverage_amount': np.random.uniform(1000, 100000, n_samples),
        'hospital_type': np.random.choice(['Public', 'Private', 'Specialty'], n_samples),
        'previous_fraud_cases': np.random.randint(0, 5, n_samples),
        'provider_claim_count': np.random.randint(10, 200, n_samples),
        'emergency_admission': np.random.choice([0, 1], n_samples),
        'outcome': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    df.to_csv(raw_data_path.path, index=False)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(train_data_path.path, index=False)
    test_df.to_csv(test_data_path.path, index=False)
    print("Ingestion step completed successfully.")

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy"]
)
def feature_engineering_step(train_in: Input[Dataset], test_in: Input[Dataset], train_out: Output[Dataset], test_out: Output[Dataset]):
    """Engineering features for training."""
    import pandas as pd
    
    def eng_features(df):
        df = df.copy()
        df['claim_to_coverage_ratio'] = df['claim_amount'] / (df['coverage_amount'] + 1)
        df['fraud_history_score'] = df['previous_fraud_cases'] / (df['provider_claim_count'] + 1)
        df['claim_frequency_score'] = df['previous_claims'] / (df['patient_age'] + 1)
        df['risk_score'] = (df['previous_fraud_cases'] * 5) + (df['emergency_admission'] * 2) + (df['claim_to_coverage_ratio'] * 10)
        return df

    tr_df = pd.read_csv(train_in.path)
    te_df = pd.read_csv(test_in.path)
    
    eng_features(tr_df).to_csv(train_out.path, index=False)
    eng_features(te_df).to_csv(test_out.path, index=False)
    print("Feature engineering completed.")

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "catboost"]
)
def train_model_step(train_data: Input[Dataset], model_output: Output[Model], preprocessor_output: Output[Artifact]):
    """Train XGBoost / CatBoost model."""
    import pandas as pd
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from catboost import CatBoostClassifier
    
    df = pd.read_csv(train_data.path)
    X = df.drop(columns=['outcome'])
    y = df['outcome']
    
    num_cols = ['patient_age', 'previous_claims', 'claim_amount', 'coverage_amount',
                'claim_to_coverage_ratio', 'fraud_history_score', 'claim_frequency_score', 'risk_score']
    cat_cols = ['gender', 'policy_type', 'hospital_type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    model = CatBoostClassifier(iterations=50, depth=5, learning_rate=0.1, verbose=0)
    model.fit(X_processed, y)
    
    joblib.dump(model, model_output.path + ".pkl")
    joblib.dump(preprocessor, preprocessor_output.path + ".pkl")
    print("Model training completed successfully.")

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "catboost"],
)
def evaluate_model_step(test_data: Input[Dataset], model_input: Input[Model], preprocessor_input: Input[Artifact], metrics_out: Output[Metrics]):
    """Evaluate performance of trained model."""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    df = pd.read_csv(test_data.path)
    X = df.drop(columns=['outcome'])
    y = df['outcome']
    
    model = joblib.load(model_input.path + ".pkl")
    preprocessor = joblib.load(preprocessor_input.path + ".pkl")
    
    X_processed = preprocessor.transform(X)
    preds = model.predict(X_processed)
    
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    metrics_out.log_metric("accuracy", float(acc))
    metrics_out.log_metric("precision", float(prec))
    metrics_out.log_metric("recall", float(rec))
    metrics_out.log_metric("f1_score", float(f1))
    print("Evaluation completed successfully.")

@dsl.pipeline(
    name="medical-claim-fraud-detection-pipeline",
    description="End-to-end medical claim fraud detection ML training pipeline on Kubeflow."
)
def training_pipeline():
    ingest = ingest_data_step()
    
    fe = feature_engineering_step(
        train_in=ingest.outputs['train_data_path'],
        test_in=ingest.outputs['test_data_path']
    )
    
    train = train_model_step(
        train_data=fe.outputs['train_out']
    )
    
    evaluate = evaluate_model_step(
        test_data=fe.outputs['test_out'],
        model_input=train.outputs['model_output'],
        preprocessor_input=train.outputs['preprocessor_output']
    )

if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(training_pipeline, 'medical_claim_pipeline.yaml')
    print("Kubeflow Pipeline compiled to 'medical_claim_pipeline.yaml'")
