import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import joblib

class DebugInteractiveCropModel:
    """Debug version to see what's happening with feature engineering"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        
    def load_trained_model(self):
        """Load the best trained model"""
        print("Loading trained model...")
        
        # Get the project root directory
        project_root = Path.cwd()
        
        model_path = project_root / "models" / "comprehensive_final_model.joblib"
        if not model_path.exists():
            print("❌ No trained model found!")
            return False
        
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path.name}")
            
            # Load label encoder
            encoder_path = model_path.parent / f"{model_path.stem.replace('_model', '_label_encoder')}.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print("✅ Label encoder loaded")
            
            # Load feature list
            features_path = model_path.parent / f"{model_path.stem.replace('_model', '_features')}.joblib"
            if features_path.exists():
                self.feature_cols = joblib.load(features_path)
                print("✅ Feature list loaded")
                print(f"Expected features: {len(self.feature_cols)}")
                print(f"Feature names: {self.feature_cols}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def engineer_features(self, input_data):
        """Create engineered features for prediction"""
        print("\n🔧 ENGINEERING FEATURES...")
        df = pd.DataFrame([input_data])
        print(f"Initial input shape: {df.shape}")
        print(f"Initial columns: {list(df.columns)}")
        
        # 1. Core nutrient interaction features
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorus'] + 1e-8)
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-8)
        df['P_K_ratio'] = df['Phosphorus'] / (df['Potassium'] + 1e-8)
        
        # 2. Nutrient balance
        df['nutrient_sum'] = df['Nitrogen'] + df['Phosphorus'] + df['Potassium']
        df['nutrient_balance'] = df['Nitrogen'] / (df['nutrient_sum'] + 1e-8)
        
        # 3. Climate interaction features
        df['temp_humidity_index'] = df['Temperature'] * df['Humidity'] / 100
        df['rainfall_temp_ratio'] = df['Rainfall'] / (df['Temperature'] + 1e-8)
        
        # 4. pH-based features
        df['ph_optimal'] = np.where((df['pH_Value'] >= 6.0) & (df['pH_Value'] <= 7.5), 1, 0)
        
        # 5. Growing condition features
        df['growing_degree_days'] = np.where(df['Temperature'] > 10, df['Temperature'] - 10, 0)
        df['moisture_index'] = df['Humidity'] * df['Rainfall'] / 1000
        
        # 6. Soil fertility indicators
        df['soil_fertility_score'] = (df['Nitrogen'] + df['Phosphorus'] + df['Potassium']) / 3
        
        # 7. Seasonal indicators
        def classify_season(temp, rainfall):
            if temp > 25 and rainfall > 150:
                return 'Monsoon'
            elif 15 <= temp <= 25 and rainfall < 100:
                return 'Winter'
            elif temp > 30 and rainfall < 80:
                return 'Summer'
            else:
                return 'Mixed'
        
        df['growing_season'] = df.apply(
            lambda x: classify_season(x['Temperature'], x['Rainfall']), axis=1
        )
        
        # 8. Crop-specific suitability features
        df['rice_suitability'] = np.where(
            (df['Temperature'] >= 20) & (df['Temperature'] <= 35) &
            (df['Humidity'] >= 70) & (df['Rainfall'] >= 150), 1, 0
        )
        
        df['wheat_suitability'] = np.where(
            (df['Temperature'] >= 15) & (df['Temperature'] <= 25) &
            (df['pH_Value'] >= 6.0) & (df['pH_Value'] <= 7.5), 1, 0
        )
        
        df['cotton_suitability'] = np.where(
            (df['Temperature'] >= 25) & (df['Temperature'] <= 35) &
            (df['Rainfall'] >= 60) & (df['Rainfall'] <= 120), 1, 0
        )
        
        # 9. Boundary condition features
        df['extreme_temp'] = np.where((df['Temperature'] < 15) | (df['Temperature'] > 35), 1, 0)
        df['extreme_ph'] = np.where((df['pH_Value'] < 5.5) | (df['pH_Value'] > 8.0), 1, 0)
        df['extreme_rainfall'] = np.where((df['Rainfall'] < 50) | (df['Rainfall'] > 300), 1, 0)
        
        print(f"After engineering shape: {df.shape}")
        print(f"Engineered columns: {list(df.columns)}")
        
        # Debug: Check values before returning
        print(f"\n🔍 DEBUG: Values before return:")
        for col in df.columns:
            print(f"  {col}: {df[col].values[0]} (type: {type(df[col].values[0])})")
        
        return df
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        print("\n🔧 PREPROCESSING INPUT...")
        
        # Engineer features
        df_eng = self.engineer_features(input_data)
        
        # Ensure all required features are present
        if self.feature_cols:
            missing_features = [col for col in self.feature_cols if col not in df_eng.columns]
            print(f"Missing features: {missing_features}")
            for feature in missing_features:
                df_eng[feature] = 0  # Default value for missing features
                print(f"Added missing feature: {feature} = 0")
        
        # Handle categorical features (only truly categorical ones)
        categorical_features = []
        for i, col in enumerate(df_eng.columns):
            if df_eng[col].dtype == 'object':  # Only convert string/object columns
                categorical_features.append(i)
        
        print(f"Categorical features: {categorical_features}")
        
        # Convert categorical features to numerical
        X = df_eng.values.copy()
        print(f"\n🔍 DEBUG: Before categorical conversion:")
        print(f"  growing_season value: {X[0, 18]} (type: {type(X[0, 18])})")
        
        for i in categorical_features:
            unique_vals = np.unique(X[:, i])
            val_to_num = {val: idx for idx, val in enumerate(unique_vals)}
            X[:, i] = np.array([val_to_num[val] for val in X[:, i]])
            print(f"Converted categorical feature {i} ({df_eng.columns[i]}): {unique_vals} -> {val_to_num}")
        
        print(f"\n🔍 DEBUG: After categorical conversion:")
        print(f"  growing_season value: {X[0, 18]} (type: {type(X[0, 18])})")
        
        print(f"Final X shape: {X.shape}")
        print(f"Final X sample: {X[0]}")
        
        # Print detailed feature values
        print("\n📊 DETAILED FEATURE VALUES:")
        for i, col in enumerate(df_eng.columns):
            print(f"  {col}: {X[0, i]}")
        
        return X
    
    def predict_crop(self, input_data):
        """Predict crop recommendation"""
        if not self.model:
            return None, "Model not loaded"
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            print(f"\n🎯 MAKING PREDICTION...")
            print(f"Input to model shape: {X.shape}")
            
            # Make prediction
            if hasattr(self.model, 'named_steps'):
                # Pipeline model
                y_pred = self.model.predict(X)
                print(f"Pipeline prediction: {y_pred}")
            else:
                # Direct model
                y_pred = self.model.predict(X)
                print(f"Direct prediction: {y_pred}")
            
            # Decode prediction
            if self.label_encoder:
                crop_name = self.label_encoder.inverse_transform(y_pred)[0]
                print(f"Decoded prediction: {crop_name}")
            else:
                crop_name = y_pred[0]
                print(f"Raw prediction: {crop_name}")
            
            return crop_name, None
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, f"Prediction error: {e}"

def main():
    """Test the debug system"""
    model = DebugInteractiveCropModel()
    
    if not model.load_trained_model():
        return
    
    # Test with rice parameters
    print("\n" + "="*60)
    print("🧪 TESTING WITH RICE PARAMETERS")
    print("="*60)
    
    rice_input = {
        'Nitrogen': 90,
        'Phosphorus': 42,
        'Potassium': 43,
        'Temperature': 20.88,
        'Humidity': 82.00,
        'pH_Value': 6.50,
        'Rainfall': 202.94
    }
    
    predicted_crop, error = model.predict_crop(rice_input)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print(f"\n✅ Final Result: {predicted_crop}")

if __name__ == "__main__":
    main()
