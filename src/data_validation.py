import os
import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataValidation")

class DataValidator:
    def __init__(self, raw_path="data/raw/claims_data.csv", report_dir="reports"):
        self.raw_path = raw_path
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def run_validation(self):
        """
        Runs validation and generates data drift/quality reports using Evidently AI.
        """
        logger.info("Starting data validation checks using Evidently AI...")
        
        if not os.path.exists(self.raw_path):
            logger.error(f"Raw dataset not found at {self.raw_path}. Run data ingestion first.")
            return False

        # Load reference data
        df = pd.read_csv(self.raw_path)
        
        # Split into reference and current for drift simulation
        # In a real environment, reference would be training data, and current would be production/inference data
        reference_data = df.sample(frac=0.5, random_state=42)
        current_data = df.drop(reference_data.index)
        
        logger.info("Generating Evidently Data Drift & Quality Report...")
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(target_name="is_fraud")
        ])
        
        data_drift_report.run(reference_data=reference_data, current_data=current_data)
        
        report_path = os.path.join(self.report_dir, "evidently_drift_report.html")
        data_drift_report.save_html(report_path)
        logger.info(f"Evidently AI Drift Report saved successfully to: {report_path}")
        return True

if __name__ == "__main__":
    validator = DataValidator()
    validator.run_validation()
