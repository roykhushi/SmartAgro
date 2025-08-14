#!/usr/bin/env python3
"""
Quick Model Viewer - Simple script to show model basics
"""

import joblib
import sys
from pathlib import Path

def quick_model_info():
    """Display essential model information quickly"""
    
    print("🤖 NITROGEN MANAGEMENT MODEL - QUICK VIEW")
    print("=" * 50)
    
    try:
        # Check if model files exist
        model_path = Path('models/comprehensive_final_model.joblib')
        encoder_path = Path('models/comprehensive_final_label_encoder.joblib')
        features_path = Path('models/comprehensive_final_features.joblib')
        
        if not model_path.exists():
            print("❌ Model file not found!")
            return False
        
        # Load components
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        features = joblib.load(features_path)
        
        # Basic info
        print(f"📁 Model File: {model_path.name}")
        print(f"📦 File Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"🧠 Model Type: {type(model).__name__}")
        print(f"🎯 Features: {len(features)}")
        print(f"🌾 Crops: {len(encoder.classes_)}")
        
        # Show crops
        print(f"\n🌱 Supported Crops ({len(encoder.classes_)}):")
        for i, crop in enumerate(encoder.classes_, 1):
            print(f"  {i:2d}. {crop}")
        
        # Show key features
        print(f"\n🔧 Key Features ({len(features)}):")
        for i, feature in enumerate(features[:10], 1):  # Show first 10
            print(f"  {i:2d}. {feature}")
        if len(features) > 10:
            print(f"     ... and {len(features) - 10} more")
        
        # Performance info
        results_path = Path('models/comprehensive_final_results.txt')
        if results_path.exists():
            print(f"\n📈 Performance:")
            with open(results_path, 'r') as f:
                lines = f.readlines()
                for line in lines[4:8]:  # Show key performance lines
                    if line.strip():
                        print(f"  {line.strip()}")
        
        print("\n✅ Model loaded and ready for predictions!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_model_structure(model):
    """Show model internal structure"""
    print(f"\n🏗️  Model Structure:")
    
    if hasattr(model, 'steps'):
        print(f"  Pipeline with {len(model.steps)} steps:")
        for i, (name, step) in enumerate(model.steps, 1):
            print(f"    {i}. {name}: {type(step).__name__}")
    else:
        print(f"  Direct model: {type(model).__name__}")
    
    if hasattr(model, 'get_params'):
        key_params = {}
        for key, value in model.get_params().items():
            if any(term in key for term in ['n_estimators', 'max_depth', 'random_state']):
                key_params[key] = value
        
        if key_params:
            print(f"  Key parameters:")
            for key, value in key_params.items():
                print(f"    {key}: {value}")

if __name__ == "__main__":
    success = quick_model_info()
    
    if success and len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        try:
            model = joblib.load('models/comprehensive_final_model.joblib')
            show_model_structure(model)
        except Exception as e:
            print(f"❌ Error showing details: {e}")
    
    print(f"\n💡 For full inspection: python show_model.py")
    print(f"💡 For interaction: python src/interactive_crop_model.py")

