# 🌾 Nitrogen Management Crop Recommendation System

A machine learning system for intelligent crop recommendations and nitrogen management in agriculture.

## 🎯 Overview

This system provides farmers with AI-powered crop recommendations based on soil and climate parameters, featuring:
- **Crop Recommendations** for 24 major crops
- **Nitrogen Management** strategies  
- **Interactive Interface** for real-time predictions
- **Data Visualizations** for insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone and navigate to project
git clone <repository-url>
cd nitrogen_management

# Install dependencies
pip install -r requirements.txt

# Run the interactive system
python src/interactive_crop_model.py
```

### Example Usage
```
🌾 Welcome to the Nitrogen Management Crop Recommendation System!

Enter your soil and climate parameters:
Nitrogen (kg/ha): 90
Phosphorus (kg/ha): 65
Potassium (kg/ha): 45
Temperature (°C): 26
Humidity (%): 85
pH Value: 7.8
Rainfall (mm): 220

🌱 RECOMMENDED CROP: KidneyBeans
💧 NITROGEN STATUS: Optimal
🔄 ROTATION SUGGESTIONS: Wheat, Maize, Oilseeds
```

## 📁 Project Structure

```
nitrogen_management/
├── src/                              # Source code
│   ├── interactive_crop_model.py     # Main application
│   ├── auto_visualize.py            # Automated visualization
│   ├── data_visualizer.py           # Data visualization tools
│   └── image_generator.py           # Chart generation
├── models/                          # Trained ML models
│   ├── comprehensive_final_model.joblib
│   ├── comprehensive_final_label_encoder.joblib
│   └── comprehensive_final_features.joblib
├── data/                           # Training datasets
│   ├── sample_train.csv
│   └── sample_test.csv
├── output/                         # Results and visualizations
│   ├── sample_predictions.csv
│   ├── nitrogen_recommendations.csv
│   └── output_images/             # Generated charts
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 📊 Core Features

### Model Specifications
- **Algorithm**: Ensemble ML model
- **Training Data**: 2,400+ samples across 24 crops
- **Accuracy**: 99.33% cross-validation
- **Crops Supported**: Rice, Wheat, Maize, Cotton, Pulses, Fruits, etc.

### Visualizations
Generate comprehensive charts and insights:
```bash
# Generate all visualizations
python src/auto_visualize.py

# Individual modules
python src/data_visualizer.py
python src/image_generator.py
```

## 🌾 Supported Crops

**Cereals**: Rice, Wheat, Maize  
**Pulses**: Lentil, KidneyBeans, ChickPea  
**Fruits**: Apple, Banana, Mango, Orange  
**Commercial**: Cotton, Sugarcane, Coffee  
**Vegetables**: Muskmelon, Watermelon  

## 🔧 Additional Tools

The project includes several utility scripts:
- `check_model.py` - Model validation
- `test_model.py` - Model testing
- `show_model.py` - Model inspection
- `visualize_model.py` - Model visualization
- `quick_model_viewer.py` - Quick model overview

## 📈 Output

The system generates:
- **Crop predictions** with confidence scores
- **Nitrogen recommendations** with optimal ranges
- **Crop rotation suggestions** for sustainability
- **Visual analytics** charts and dashboards
- **Performance reports** and summaries

All outputs are saved in the `output/` directory with detailed CSV files and high-resolution images.

## 💡 Key Features

- **Nutrient Analysis**: N/P/K ratio optimization
- **Climate Intelligence**: Temperature-humidity indexing
- **Soil Health**: pH and fertility scoring
- **Agricultural Insights**: Crop-specific recommendations

## 🌍 Applications

**For Farmers**: Get data-driven crop recommendations and nitrogen management advice  
**For Researchers**: Analyze crop-environment relationships and farming patterns  
**For Extension Workers**: Provide evidence-based agricultural guidance  

---

**🌾 AI-Powered Agriculture for Sustainable Farming 🚀**