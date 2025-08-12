# src/model_utils.py -> RandomForestClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_rf(X_train, y_train, config=None):
    params = config.get('rf_params', {}) if config else {}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        'accuracy': accuracy_score(y, preds),
        'report': classification_report(y, preds)
    }

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
