import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import joblib

class InteractiveCropModel:
    """Interactive crop recommendation model for nitrogen management and crop rotation"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        self.scaler = None
        
    def load_trained_model(self):
        """Load the best trained model"""
        print("Loading trained model...")
        
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        
        model_path = project_root / "models" / "comprehensive_final_model.joblib"
        if not model_path.exists():
            model_path = project_root / "models" / "final_production_model.joblib"
            if not model_path.exists():
                model_path = project_root / "models" / "improved_high_accuracy_model.joblib"
                if not model_path.exists():
                    print("❌ No trained model found! Please train a model first.")
                    print(f"   Looking in: {project_root / 'models'}")
                    return False
        
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path.name}")
            print(f"Model Type: {type(self.model)}")
            print(f"Model Parameters: {self.model.get_params()}")
            
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
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def engineer_features(self, input_data):

        df = pd.DataFrame([input_data])
        
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
        
        return df
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Engineer features
        df_eng = self.engineer_features(input_data)
        
        # Ensure all required features are present
        if self.feature_cols:
            missing_features = [col for col in self.feature_cols if col not in df_eng.columns]
            for feature in missing_features:
                df_eng[feature] = 0  # Default value for missing features
        
        # Handle categorical features 
        categorical_features = []
        for i, col in enumerate(df_eng.columns):
            if df_eng[col].dtype == 'object':  # Only convert string/object columns
                categorical_features.append(i)
        
        # Convert categorical features to numerical
        X = df_eng.values.copy()
        for i in categorical_features:
            unique_vals = np.unique(X[:, i])
            val_to_num = {val: idx for idx, val in enumerate(unique_vals)}
            X[:, i] = np.array([val_to_num[val] for val in X[:, i]])
        
        return X
    
    def predict_crop(self, input_data):
        """Predict crop recommendation"""
        if not self.model:
            return None, "Model not loaded"
        
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction
            if hasattr(self.model, 'named_steps'):
                # Pipeline model
                y_pred = self.model.predict(X)
            else:
                # Direct model
                y_pred = self.model.predict(X)
            
            # Decode prediction
            if self.label_encoder:
                crop_name = self.label_encoder.inverse_transform(y_pred)[0]
            else:
                crop_name = y_pred[0]
            
            return crop_name, None
            
        except Exception as e:
            return None, f"Prediction error: {e}"
    
    def get_nitrogen_recommendations(self, crop_name, current_nitrogen):
        """Get nitrogen management recommendations for the predicted crop"""
        recommendations = {
            'Rice': {
                'optimal_nitrogen': (80, 120),
                'application_timing': 'Split application: 50% at transplanting, 25% at tillering, 25% at panicle initiation',
                'notes': 'High nitrogen requirement. Avoid excessive N to prevent lodging.'
            },
            'Wheat': {
                'optimal_nitrogen': (60, 100),
                'application_timing': 'Split application: 50% at sowing, 25% at crown root initiation, 25% at first node',
                'notes': 'Moderate nitrogen requirement. Apply N based on soil testing.'
            },
            'Maize': {
                'optimal_nitrogen': (70, 110),
                'application_timing': 'Split application: 30% at sowing, 40% at knee-high stage, 30% at tasseling',
                'notes': 'High nitrogen requirement. Critical for grain yield.'
            },
            'Cotton': {
                'optimal_nitrogen': (60, 100),
                'application_timing': 'Split application: 25% at sowing, 50% at square formation, 25% at flowering',
                'notes': 'Moderate nitrogen requirement. Avoid late N application.'
            },
            'Sugarcane': {
                'optimal_nitrogen': (80, 120),
                'application_timing': 'Split application: 40% at planting, 30% at tillering, 30% at grand growth',
                'notes': 'High nitrogen requirement. Apply N in 3-4 splits.'
            }
        }
        
        if crop_name in recommendations:
            rec = recommendations[crop_name]
            optimal_min, optimal_max = rec['optimal_nitrogen']
            
            if current_nitrogen < optimal_min:
                status = "⚠️ Nitrogen Deficient"
                action = f"Apply {optimal_min - current_nitrogen} kg/ha additional nitrogen"
            elif current_nitrogen > optimal_max:
                status = "⚠️ Nitrogen Excessive"
                action = "Reduce nitrogen application to prevent environmental issues"
            else:
                status = "✅ Nitrogen Optimal"
                action = "Maintain current nitrogen levels"
            
            return {
                'status': status,
                'action': action,
                'optimal_range': f"{optimal_min}-{optimal_max} kg/ha",
                'timing': rec['application_timing'],
                'notes': rec['notes']
            }
        else:
            return {
                'status': "ℹ️ General Recommendation",
                'action': "Apply nitrogen based on soil testing and crop growth stage",
                'optimal_range': "60-100 kg/ha (general guideline)",
                'timing': "Split application: 50% at sowing/transplanting, 25% at vegetative growth, 25% at reproductive stage",
                'notes': "Adjust based on soil type, climate, and crop variety."
            }
    
    def get_crop_rotation_suggestions(self, predicted_crop):
        """Get crop rotation suggestions for sustainable farming"""
        rotation_suggestions = {
            'Rice': {
                'next_crops': ['Wheat', 'Pulses', 'Oilseeds'],
                'rotation_benefits': 'Rice-wheat rotation improves soil structure and nutrient cycling',
                'nitrogen_management': 'Legumes in rotation can reduce N requirement by 20-30%'
            },
            'Wheat': {
                'next_crops': ['Pulses', 'Oilseeds', 'Maize'],
                'rotation_benefits': 'Wheat-pulse rotation improves soil fertility and breaks pest cycles',
                'nitrogen_management': 'Include legumes to naturally fix atmospheric nitrogen'
            },
            'Maize': {
                'next_crops': ['Pulses', 'Wheat', 'Oilseeds'],
                'rotation_benefits': 'Maize-pulse rotation improves soil organic matter',
                'nitrogen_management': 'Rotate with legumes to reduce synthetic N requirement'
            },
            'Cotton': {
                'next_crops': ['Wheat', 'Pulses', 'Rice'],
                'rotation_benefits': 'Cotton-wheat rotation helps control cotton pests',
                'nitrogen_management': 'Include green manure crops to improve soil N'
            },
            'Pulses': {
                'next_crops': ['Wheat', 'Rice', 'Maize'],
                'rotation_benefits': 'Pulses improve soil nitrogen through biological fixation',
                'nitrogen_management': 'Reduce N application by 20-30% after pulse crops'
            }
        }
        
        if predicted_crop in rotation_suggestions:
            return rotation_suggestions[predicted_crop]
        else:
            return {
                'next_crops': ['Wheat', 'Pulses', 'Oilseeds'],
                'rotation_benefits': 'Diversified rotation improves soil health and reduces pest pressure',
                'nitrogen_management': 'Include legumes and green manure crops for sustainable N management'
            }
    
    def interactive_prediction(self):
        """Interactive crop prediction interface"""
        print("\n🌾 INTERACTIVE CROP RECOMMENDATION SYSTEM 🌾")
        print("=" * 60)
        print("This system helps farmers with:")
        print("• Crop recommendations based on soil and climate conditions")
        print("• Nitrogen management strategies")
        print("• Crop rotation planning")
        print("=" * 60)
        
        # Load model
        if not self.load_trained_model():
            print("\n❌ Cannot proceed without a trained model.")
            return
        
        while True:
            print("\n" + "="*60)
            print("📊 ENTER SOIL AND CLIMATE PARAMETERS")
            print("="*60)
            
            try:
                # Get user input
                nitrogen = float(input("Enter Nitrogen content (kg/ha): "))
                phosphorus = float(input("Enter Phosphorus content (kg/ha): "))
                potassium = float(input("Enter Potassium content (kg/ha): "))
                temperature = float(input("Enter Temperature (°C): "))
                humidity = float(input("Enter Humidity (%): "))
                ph_value = float(input("Enter pH Value: "))
                rainfall = float(input("Enter Rainfall (mm): "))
                
                # Create input data
                input_data = {
                    'Nitrogen': nitrogen,
                    'Phosphorus': phosphorus,
                    'Potassium': potassium,
                    'Temperature': temperature,
                    'Humidity': humidity,
                    'pH_Value': ph_value,
                    'Rainfall': rainfall
                }
                
                print(f"\n🔍 Analyzing conditions...")
                print(f"   Soil: N={nitrogen}, P={phosphorus}, K={potassium} kg/ha")
                print(f"   Climate: Temp={temperature}°C, Humidity={humidity}%, Rainfall={rainfall}mm")
                print(f"   pH: {ph_value}")
                
                # Predict crop
                predicted_crop, error = self.predict_crop(input_data)
                
                if error:
                    print(f"❌ Error: {error}")
                    continue
                
                print(f"\n🌱 RECOMMENDED CROP: {predicted_crop.upper()}")
                print("=" * 60)
                
                # Get nitrogen recommendations
                n_rec = self.get_nitrogen_recommendations(predicted_crop, nitrogen)
                print(f"💧 NITROGEN MANAGEMENT:")
                print(f"   Status: {n_rec['status']}")
                print(f"   Action: {n_rec['action']}")
                print(f"   Optimal Range: {n_rec['optimal_range']}")
                print(f"   Application Timing: {n_rec['timing']}")
                print(f"   Notes: {n_rec['notes']}")
                
                # Get rotation suggestions
                rotation = self.get_crop_rotation_suggestions(predicted_crop)
                print(f"\n🔄 CROP ROTATION SUGGESTIONS:")
                print(f"   Next Crops: {', '.join(rotation['next_crops'])}")
                print(f"   Benefits: {rotation['rotation_benefits']}")
                print(f"   N Management: {rotation['nitrogen_management']}")
                
                # Additional recommendations
                print(f"\n💡 ADDITIONAL RECOMMENDATIONS:")
                if ph_value < 6.0:
                    print("   • Consider liming to raise soil pH for better nutrient availability")
                elif ph_value > 7.5:
                    print("   • Monitor micronutrient availability at high pH")
                
                if rainfall < 100:
                    print("   • Ensure adequate irrigation for optimal crop growth")
                elif rainfall > 300:
                    print("   • Monitor for waterlogging and adjust drainage")
                
                if temperature < 15:
                    print("   • Consider cold-tolerant crop varieties")
                elif temperature > 35:
                    print("   • Implement heat stress management strategies")
                
            except ValueError:
                print("❌ Please enter valid numbers for all parameters.")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using the Crop Recommendation System!")
                break
            
            # Ask if user wants to continue
            print(f"\n" + "="*60)
            choice = input("🔄 Make another prediction? (y/n): ").lower().strip()
            if choice not in ['y', 'yes', '']:
                print("\n👋 Thank you for using the Crop Recommendation System!")
                break

def main():
    """Run the interactive crop recommendation system"""
    model = InteractiveCropModel()
    model.interactive_prediction()

if __name__ == "__main__":
    main()
