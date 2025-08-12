import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

FEATURES = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    X = df[FEATURES]
    y = df['Crop']

    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    X_proc = num_pipeline.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X_proc, y_enc, num_pipeline, le
