import joblib

# Load the model
model = joblib.load('models/comprehensive_final_model.joblib')

print("🤖 MODEL ARCHITECTURE & ALGORITHM DETAILS")
print("=" * 60)

print(f"Model type: {type(model)}")
print(f"Pipeline steps: {len(model.steps)}")

print("\n📋 PIPELINE DETAILS:")
for i, (name, step) in enumerate(model.steps):
    print(f"  Step {i+1}: {name} - {type(step).__name__}")
    
    # Check specific attributes for each step
    if hasattr(step, 'strategy'):
        print(f"    Strategy: {step.strategy}")
    if hasattr(step, 'n_features_in_'):
        print(f"    Features: {step.n_features_in_}")
    if hasattr(step, 'n_estimators'):
        print(f"    Estimators: {step.n_estimators}")
    if hasattr(step, 'max_depth'):
        print(f"    Max depth: {step.max_depth}")
    if hasattr(step, 'min_samples_split'):
        print(f"    Min samples split: {step.min_samples_split}")
    if hasattr(step, 'min_samples_leaf'):
        print(f"    Min samples leaf: {step.min_samples_leaf}")
    if hasattr(step, 'random_state'):
        print(f"    Random state: {step.random_state}")
    if hasattr(step, 'class_weight'):
        print(f"    Class weight: {step.class_weight}")

print("\n🔍 MODEL PARAMETERS:")
if hasattr(model, 'get_params'):
    params = model.get_params()
    for key, value in params.items():
        if 'randomforest' in key.lower() or 'estimator' in key.lower():
            print(f"  {key}: {value}")

print("\n📊 FEATURE INFORMATION:")
features = joblib.load('models/comprehensive_final_features.joblib')
print(f"  Total features: {len(features)}")
print(f"  Feature names: {features}")

print("\n🎯 LABEL ENCODER:")
encoder = joblib.load('models/comprehensive_final_label_encoder.joblib')
print(f"  Classes: {encoder.classes_}")
print(f"  Number of classes: {len(encoder.classes_)}")

