#!/usr/bin/env python3
"""
Auto-visualization script for the Nitrogen Management System
This script automatically generates visualizations whenever the system runs
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from image_generator import ImageGenerator
from data_visualizer import DataVisualizer

def auto_generate_visualizations():
    """Automatically generate all visualizations"""
    print("🖼️  Auto-generating visualizations...")
    
    try:
        # First generate sample visualizations
        print("📊 Generating sample visualizations...")
        sample_generator = ImageGenerator()
        sample_generator.create_sample_data_visualizations()
        
        # Then generate data-based visualizations
        print("📈 Generating data-based visualizations...")
        data_visualizer = DataVisualizer()
        data_visualizer.generate_all_visualizations()
        
        print("✅ All visualizations generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return False

def quick_visualization():
    """Generate quick visualizations for common use cases"""
    print("⚡ Generating quick visualizations...")
    
    try:
        # Quick feature importance
        generator = ImageGenerator()
        feature_importance = {
            'Nitrogen': 0.25,
            'Phosphorus': 0.20,
            'Potassium': 0.18,
            'Temperature': 0.15,
            'Humidity': 0.12,
            'pH_Value': 0.08,
            'Rainfall': 0.02
        }
        generator.create_feature_importance_plot(feature_importance)
        
        # Quick crop distribution
        crop_data = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Pulses'] * 20
        generator.create_crop_distribution_plot(crop_data)
        
        print("✅ Quick visualizations generated!")
        return True
        
    except Exception as e:
        print(f"❌ Error in quick visualization: {e}")
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_visualization()
    else:
        auto_generate_visualizations()
