import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Preprocessing")

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to cap outliers at specific percentiles.
    Supports both pandas DataFrames and numpy arrays.
    """
    def __init__(self, columns=None, lower_percentile=1.0, upper_percentile=99.0):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.caps_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in self.columns:
                if col in X.columns:
                    lower = np.percentile(X[col].dropna(), self.lower_percentile)
                    upper = np.percentile(X[col].dropna(), self.upper_percentile)
                    self.caps_[col] = (lower, upper)
        else:
            # Handle numpy array input
            num_cols = X.shape[1]
            for col_idx in range(num_cols):
                col_data = X[:, col_idx]
                # Filter out NaNs if any
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    lower = np.percentile(valid_data, self.lower_percentile)
                    upper = np.percentile(valid_data, self.upper_percentile)
                else:
                    lower, upper = -np.inf, np.inf
                self.caps_[col_idx] = (lower, upper)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            for col, (lower, upper) in self.caps_.items():
                if col in X_copy.columns:
                    X_copy[col] = np.clip(X_copy[col], lower, upper)
            return X_copy
        else:
            # Handle numpy array input
            X_copy = X.copy()
            for col_idx, (lower, upper) in self.caps_.items():
                X_copy[:, col_idx] = np.clip(X_copy[:, col_idx], lower, upper)
            return X_copy



class DataPreprocessor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipeline = None
        self.feature_names = None

    def build_pipeline(self, num_cols, cat_cols):
        """
        Builds the Scikit-Learn preprocessing pipeline.
        """
        # Numerical transformer
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outliers', OutlierCapper(columns=num_cols)),
            ('scaler', StandardScaler())
        ])

        # Categorical transformer (handles unseen values by ignoring them)
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine preprocessing steps
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )
        return self.pipeline

    def fit(self, df, num_cols, cat_cols):
        """
        Fits the pipeline on the input DataFrame.
        """
        logger.info("Fitting the preprocessing pipeline...")
        self.build_pipeline(num_cols, cat_cols)
        self.pipeline.fit(df)
        
        # Extract feature names after transformation
        # To handle modern sklearn version compatibility
        try:
            onehot_features = self.pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
            self.feature_names = list(num_cols) + list(onehot_features)
        except Exception as e:
            logger.warning(f"Could not retrieve feature names out: {e}")
            self.feature_names = num_cols + cat_cols
            
        return self

    def transform(self, df):
        """
        Transforms the input DataFrame using the fitted pipeline.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet. Please call fit() first.")
        
        transformed_array = self.pipeline.transform(df)
        # Reconstruct DataFrame with feature names
        try:
            return pd.DataFrame(transformed_array, columns=self.feature_names)
        except Exception:
            return pd.DataFrame(transformed_array)

    def fit_transform(self, df, num_cols, cat_cols):
        self.fit(df, num_cols, cat_cols)
        return self.transform(df)

    def save(self, pipeline_path=None, encoder_path=None, scaler_path=None):
        """
        Saves preprocessor components.
        """
        if pipeline_path is None:
            pipeline_path = os.path.join(self.model_dir, "preprocessor_pipeline.pkl")
        
        joblib.dump(self.pipeline, pipeline_path)
        logger.info(f"Saved preprocessing pipeline to {pipeline_path}")
        
        # Save feature names too
        names_path = os.path.join(self.model_dir, "feature_names.pkl")
        joblib.dump(self.feature_names, names_path)
        logger.info(f"Saved feature names to {names_path}")

        # Also save legacy file names for compatibility with requirements (encoder.pkl / scaler.pkl)
        # We can extract and save them separately
        if self.pipeline:
            # Save encoder
            encoder = self.pipeline.named_transformers_['cat'].named_steps['onehot']
            enc_p = encoder_path or os.path.join(self.model_dir, "encoder.pkl")
            joblib.dump(encoder, enc_p)
            
            # Save scaler
            scaler = self.pipeline.named_transformers_['num'].named_steps['scaler']
            scl_p = scaler_path or os.path.join(self.model_dir, "scaler.pkl")
            joblib.dump(scaler, scl_p)

    def load(self, pipeline_path=None):
        """
        Loads the preprocessor pipeline.
        """
        if pipeline_path is None:
            pipeline_path = os.path.join(self.model_dir, "preprocessor_pipeline.pkl")
        
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"No pipeline file found at {pipeline_path}")
            
        self.pipeline = joblib.load(pipeline_path)
        
        names_path = os.path.join(self.model_dir, "feature_names.pkl")
        if os.path.exists(names_path):
            self.feature_names = joblib.load(names_path)
            
        logger.info(f"Loaded preprocessing pipeline from {pipeline_path}")
        return self


if __name__ == "__main__":
    # Test preprocessor on synthetic data if it exists
    raw_path = "data/raw/claims_data.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        num_cols = ['patient_age', 'policy_tenure', 'claim_amount', 'length_of_stay', 
                    'provider_claim_count', 'previous_fraud_cases', 'previous_claims', 
                    'coverage_amount', 'deductible_amount']
        cat_cols = ['gender', 'policy_type', 'hospital_type', 'procedure_code', 'diagnosis_code', 'provider_id']
        
        preprocessor = DataPreprocessor()
        transformed_df = preprocessor.fit_transform(df, num_cols, cat_cols)
        preprocessor.save()
        logger.info(f"Transformed shape: {transformed_df.shape}")
    else:
        logger.warning(f"Raw data not found at {raw_path}. Run data_ingestion.py first.")
