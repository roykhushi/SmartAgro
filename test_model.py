import joblib
import pandas as pd
import numpy as np

# Load the model and components
model = joblib.load('models/comprehensive_final_model.joblib')
encoder = joblib.load('models/comprehensive_final_label_encoder.joblib')
features = joblib.load('models/comprehensive_final_features.joblib')

print("Model loaded successfully!")
print(f"Expected features: {len(features)}")
print(f"Feature names: {features}")

# Test with rice parameters from training data
rice_params = [90, 42, 43, 20.88, 82.00, 6.50, 202.94]

# Create engineered features manually (like the interactive system does)
df = pd.DataFrame([{
    'Nitrogen': 90, 'Phosphorus': 42, 'Potassium': 43,
    'Temperature': 20.88, 'Humidity': 82.00, 'pH_Value': 6.50, 'Rainfall': 202.94
}])

# Engineer features
df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorus'] + 1e-8)
df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-8)
df['P_K_ratio'] = df['Phosphorus'] / (df['Potassium'] + 1e-8)
df['nutrient_sum'] = df['Nitrogen'] + df['Phosphorus'] + df['Potassium']
df['nutrient_balance'] = df['Nitrogen'] / (df['nutrient_sum'] + 1e-8)
df['temp_humidity_index'] = df['Temperature'] * df['Humidity'] / 100
df['rainfall_temp_ratio'] = df['Rainfall'] / (df['Temperature'] + 1e-8)
df['ph_optimal'] = np.where((df['pH_Value'] >= 6.0) & (df['pH_Value'] <= 7.5), 1, 0)
df['growing_degree_days'] = np.where(df['Temperature'] > 10, df['Temperature'] - 10, 0)
df['moisture_index'] = df['Humidity'] * df['Rainfall'] / 1000
df['soil_fertility_score'] = (df['Nitrogen'] + df['Phosphorus'] + df['Potassium']) / 3

# Seasonal classification
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

# Crop suitability features
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

# Boundary conditions
df['extreme_temp'] = np.where((df['Temperature'] < 15) | (df['Temperature'] > 35), 1, 0)
df['extreme_ph'] = np.where((df['pH_Value'] < 5.5) | (df['pH_Value'] > 8.0), 1, 0)
df['extreme_rainfall'] = np.where((df['Rainfall'] < 50) | (df['Rainfall'] > 300), 1, 0)

print(f"\nEngineered features shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert to numpy array
X = df.values
print(f"Input array shape: {X.shape}")

# Make prediction
try:
    prediction = model.predict(X)
    print(f"Raw prediction: {prediction}")
    
    decoded = encoder.inverse_transform(prediction)
    print(f"Decoded prediction: {decoded}")
    
except Exception as e:
    print(f"Prediction error: {e}")
    print(f"Model expects {model.named_steps['imputer'].n_features_in_} features")

