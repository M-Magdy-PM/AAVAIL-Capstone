import unittest
import os
import pandas as pd
from cslib import fetch_data, fetch_ts, engineer_features, resolve_path
from model import model_load, model_predict, model_train
from app import app

class AavailWorkflowTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_dir = resolve_path("cs-train")
        cls.client = app.test_client()
        
    def test_01_fetch_data(self):
        """Test data ingestion from JSON files"""
        df = fetch_data(self.data_dir)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)
        
    def test_02_fetch_ts(self):
        """Test time-series conversion and caching"""
        ts_data = fetch_ts(self.data_dir)
        self.assertIsInstance(ts_data, dict)
        self.assertIn('all', ts_data)
        self.assertIn('united_kingdom', ts_data)
        
    def test_03_engineer_features(self):
        """Test feature engineering logic"""
        ts_data = fetch_ts(self.data_dir)
        df_all = ts_data['all']
        X, y, dates = engineer_features(df_all, training=True)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], dates.shape[0])
        self.assertGreater(X.shape[1], 0)
        
    def test_04_model_prediction(self):
        """Test model loading and prediction function"""
        try:
            all_data, all_models = model_load(data_dir=self.data_dir, training=False)
            self.assertIn('all', all_models)
            
            # Run prediction
            result = model_predict('all', '2018', '01', '05', all_models=all_models, test=True)
            self.assertIn('y_pred', result)
            self.assertIsNotNone(result['y_pred'])
        except Exception as e:
            self.fail(f"Model loading or prediction failed: {e}")
            
    def test_05_api_index(self):
        """Test Flask API home endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'online')
        
    def test_06_api_predict(self):
        """Test Flask API prediction endpoint"""
        response = self.client.get("/predict?country=all&year=2018&month=01&day=05&test=true")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data)

    def test_07_api_logs(self):
        """Test Flask API logs endpoint"""
        response = self.client.get("/logs/train?test=true")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('logs', data)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
