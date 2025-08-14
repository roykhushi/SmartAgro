#!/usr/bin/env python3


import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def display_header():
    """Display header information"""
    print("🤖 NITROGEN MANAGEMENT CROP RECOMMENDATION MODEL")
    print("=" * 70)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def load_and_inspect_model():
    """Load and inspect the trained model"""
    try:
        # Load model components
        model = joblib.load('models/comprehensive_final_model.joblib')
        encoder = joblib.load('models/comprehensive_final_label_encoder.joblib')
        features = joblib.load('models/comprehensive_final_features.joblib')
        
        print("\n✅ MODEL LOADING STATUS")
        print("-" * 40)
        print("✓ Main model loaded successfully")
        print("✓ Label encoder loaded successfully")
        print("✓ Feature list loaded successfully")
        
        return model, encoder, features
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def display_model_architecture(model):
    """Display model architecture details"""
    print("\n🏗️  MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"Model Type: {type(model).__name__}")
    
    if hasattr(model, 'steps'):
        print(f"Pipeline Steps: {len(model.steps)}")
        print("\n📋 Pipeline Components:")
        for i, (name, step) in enumerate(model.steps):
            print(f"  {i+1}. {name}: {type(step).__name__}")
            
            # Show step details
            if hasattr(step, 'n_estimators'):
                print(f"     └─ Number of estimators: {step.n_estimators}")
            if hasattr(step, 'max_depth'):
                print(f"     └─ Max depth: {step.max_depth}")
            if hasattr(step, 'random_state'):
                print(f"     └─ Random state: {step.random_state}")
            if hasattr(step, 'n_features_in_'):
                print(f"     └─ Input features: {step.n_features_in_}")
    else:
        print("Direct model (not a pipeline)")
        if hasattr(model, 'n_estimators'):
            print(f"Number of estimators: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"Max depth: {model.max_depth}")

def display_model_parameters(model):
    """Display model parameters"""
    print("\n⚙️  MODEL PARAMETERS")
    print("-" * 40)
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        
        # Group parameters by category
        important_params = {}
        for key, value in params.items():
            if any(term in key.lower() for term in ['estimator', 'depth', 'samples', 'features', 'random']):
                important_params[key] = value
        
        if important_params:
            for key, value in sorted(important_params.items()):
                print(f"  {key}: {value}")
        else:
            print("  No specific parameters to display")

def display_feature_information(features):
    """Display feature information"""
    print("\n🎯 FEATURE INFORMATION")
    print("-" * 40)
    print(f"Total Features: {len(features)}")
    print("\nFeature Categories:")
    
    # Categorize features
    categories = {
        "🌱 Basic Soil/Climate": [],
        "🔢 Nutrient Ratios": [],
        "🌡️  Climate Indices": [],
        "📊 Soil Health": [],
        "🌾 Crop Suitability": [],
        "⚠️  Boundary Conditions": [],
        "🗓️  Seasonal": []
    }
    
    for feature in features:
        if feature in ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']:
            categories["🌱 Basic Soil/Climate"].append(feature)
        elif 'ratio' in feature.lower():
            categories["🔢 Nutrient Ratios"].append(feature)
        elif any(term in feature.lower() for term in ['temp', 'humidity', 'rainfall']):
            categories["🌡️  Climate Indices"].append(feature)
        elif any(term in feature.lower() for term in ['fertility', 'ph_optimal', 'moisture', 'nutrient']):
            categories["📊 Soil Health"].append(feature)
        elif 'suitability' in feature.lower():
            categories["🌾 Crop Suitability"].append(feature)
        elif 'extreme' in feature.lower():
            categories["⚠️  Boundary Conditions"].append(feature)
        elif any(term in feature.lower() for term in ['season', 'growing']):
            categories["🗓️  Seasonal"].append(feature)
    
    for category, feature_list in categories.items():
        if feature_list:
            print(f"\n  {category}:")
            for feature in feature_list:
                print(f"    • {feature}")

def display_crop_information(encoder):
    """Display supported crop information"""
    print("\n🌾 SUPPORTED CROPS")
    print("-" * 40)
    crops = encoder.classes_
    print(f"Total Crops: {len(crops)}")
    
    # Categorize crops
    crop_categories = {
        "🌾 Cereals": ['Rice', 'Wheat', 'Maize'],
        "🫘 Pulses": ['Lentil', 'KidneyBeans', 'ChickPea'],
        "🍎 Fruits": ['Apple', 'Banana', 'Mango', 'Orange', 'Grapes'],
        "🏭 Commercial": ['Cotton', 'Sugarcane', 'Coffee'],
        "🥒 Vegetables": ['Muskmelon', 'Watermelon'],
        "🌿 Others": []
    }
    
    # Assign crops to categories
    assigned_crops = set()
    for category, crop_list in crop_categories.items():
        found_crops = [crop for crop in crops if crop in crop_list]
        if found_crops:
            print(f"\n  {category}:")
            for crop in found_crops:
                print(f"    • {crop}")
                assigned_crops.add(crop)
    
    # Handle unassigned crops
    unassigned = [crop for crop in crops if crop not in assigned_crops]
    if unassigned:
        crop_categories["🌿 Others"] = unassigned
        print(f"\n  🌿 Others:")
        for crop in unassigned:
            print(f"    • {crop}")

def display_performance_metrics():
    """Display performance metrics from results file"""
    print("\n📈 PERFORMANCE METRICS")
    print("-" * 40)
    
    results_file = Path('models/comprehensive_final_results.txt')
    if results_file.exists():
        with open(results_file, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("Performance metrics file not found")

def demonstrate_prediction():
    """Demonstrate model prediction with example"""
    print("\n🔮 PREDICTION DEMONSTRATION")
    print("-" * 40)
    
    try:
        model = joblib.load('models/comprehensive_final_model.joblib')
        encoder = joblib.load('models/comprehensive_final_label_encoder.joblib')
        
        # Example input (Rice-favorable conditions)
        example_input = {
            'Nitrogen': 90, 'Phosphorus': 42, 'Potassium': 43,
            'Temperature': 28, 'Humidity': 82, 'pH_Value': 6.5, 'Rainfall': 200
        }
        
        print("Example Input:")
        for key, value in example_input.items():
            print(f"  {key}: {value}")
        
        # Create simple feature set (without full engineering for demo)
        input_array = np.array(list(example_input.values())).reshape(1, -1)
        
        # Note: This is simplified - the real system does extensive feature engineering
        print("\n⚠️  Note: This is a simplified prediction demo.")
        print("The actual system performs extensive feature engineering (26 features total)")
        
    except Exception as e:
        print(f"❌ Error in prediction demo: {e}")

def generate_model_summary_file():
    """Generate a comprehensive model summary file"""
    print("\n💾 GENERATING MODEL SUMMARY FILE")
    print("-" * 40)
    
    try:
        model, encoder, features = load_and_inspect_model()
        if not model:
            return
        
        summary = {
            "model_info": {
                "type": type(model).__name__,
                "file_size_mb": round(Path('models/comprehensive_final_model.joblib').stat().st_size / (1024*1024), 2),
                "creation_date": datetime.now().isoformat()
            },
            "features": {
                "total_count": len(features),
                "feature_names": features
            },
            "crops": {
                "total_count": len(encoder.classes_),
                "supported_crops": encoder.classes_.tolist()
            }
        }
        
        # Save summary
        with open('output/model_inspection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Model summary saved to: output/model_inspection_summary.json")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")

def main():
    """Main function to run complete model inspection"""
    display_header()
    
    # Load model
    model, encoder, features = load_and_inspect_model()
    if not model:
        return
    
    # Display all information
    display_model_architecture(model)
    display_model_parameters(model)
    display_feature_information(features)
    display_crop_information(encoder)
    display_performance_metrics()
    demonstrate_prediction()
    generate_model_summary_file()
    
    print("\n" + "=" * 70)
    print("🎯 MODEL INSPECTION COMPLETE")
    print("=" * 70)
    print("\n💡 To interact with the model:")
    print("   python src/interactive_crop_model.py")
    print("\n📊 To test the model:")
    print("   python test_model.py")

if __name__ == "__main__":
    main()
