# 🌾 Nitrogen Management Crop Recommendation System

A comprehensive machine learning system for intelligent crop recommendations, nitrogen management, and sustainable crop rotation planning in agriculture.

## 🎯 Project Overview

This system combines advanced machine learning with agricultural domain knowledge to provide farmers with:
- **Crop Recommendations** based on soil and climate parameters
- **Nitrogen Management** strategies with optimal application timing
- **Crop Rotation** planning for sustainable farming
- **Interactive Interface** for real-time recommendations

## 🚀 Features

### 🌱 Core Capabilities
- **24 Major Crops** covered (Rice, Wheat, Maize, Cotton, Sugarcane, etc.)
- **Ensemble ML Model** (Random Forest + Extra Trees + Gradient Boosting)
- **26 Engineered Features** including nutrient ratios, climate indices, and soil health indicators
- **99.33% Cross-Validation Accuracy** with balanced regularization

### 💧 Nitrogen Management
- Optimal nitrogen ranges for each crop
- Split application timing recommendations
- Deficiency/excess detection and correction
- Crop-specific N requirements and considerations

### 🔄 Crop Rotation
- Sustainable farming sequences
- Nitrogen cycling optimization
- Pest and disease management strategies
- Soil health improvement recommendations

## 📁 Project Structure

```
nitrogen_management/
├── 📊 data/                          # Training and test datasets
│   ├── sample_train.csv              # Training data (2,200 samples)
│   └── sample_test.csv               # Test data (48 samples)
├── 🤖 models/                        # Trained ML models
│   ├── comprehensive_final_model.joblib
│   ├── comprehensive_final_label_encoder.joblib
│   ├── comprehensive_final_features.joblib
│   └── comprehensive_final_results.txt
├── 💻 src/                           # Source code
│   ├── interactive_crop_model.py     # Main interactive application
│   ├── model_utils.py                # Model utility functions
│   ├── train.py                      # Model training script
│   └── config.yaml                   # Configuration file
├── 📈 output/                        # Sample outputs and results
│   ├── sample_predictions.csv        # Example predictions
│   ├── nitrogen_recommendations.csv   # N management guide
│   ├── crop_rotation_examples.csv    # Rotation sequences
│   ├── model_performance_summary.txt # System capabilities
│   ├── system_demonstration.txt      # Usage examples
│   └── output_images/                # Visualization placeholders
├── 📚 docs/                          # Documentation
├── 📓 notebooks/                     # Jupyter notebooks
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd nitrogen_management
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import pandas, sklearn, joblib; print('✅ All dependencies installed successfully!')"
```

## 🚀 Quick Start

### Interactive Crop Recommendation
```bash
cd src
python interactive_crop_model.py
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

## 📊 Model Training

### Train New Model
```bash
cd src
python train.py
```

### Model Specifications
- **Algorithm**: Ensemble (Random Forest + Extra Trees + Gradient Boosting)
- **Training Data**: 2,400 samples across 24 crop types
- **Features**: 26 engineered features
- **Cross-Validation**: 5-fold stratified
- **Regularization**: Balanced to prevent overfitting

## 🌾 Supported Crops

### Cereals
- Rice, Wheat, Maize

### Pulses
- Lentil, KidneyBeans, ChickPea

### Fruits
- Apple, Banana, Mango, Orange

### Commercial Crops
- Cotton, Sugarcane, Coffee

### Vegetables
- Muskmelon, Watermelon

## 💡 Key Features Explained

### 1. Nutrient Interactions
- **N/P Ratio**: Nitrogen to Phosphorus balance
- **N/K Ratio**: Nitrogen to Potassium balance
- **P/K Ratio**: Phosphorus to Potassium balance

### 2. Climate Indices
- **Temperature-Humidity Index**: Combined climate stress
- **Rainfall-Temperature Ratio**: Water availability
- **Growing Degree Days**: Growing season length

### 3. Soil Health Indicators
- **pH Optimal**: pH suitability scoring
- **Soil Fertility Score**: Overall soil health
- **Moisture Index**: Water retention capacity

### 4. Agricultural Intelligence
- **Crop-Specific Suitability**: Tailored recommendations
- **Seasonal Classifications**: Growing season optimization
- **Boundary Conditions**: Extreme condition handling

## 🔧 Configuration

Edit `src/config.yaml` to customize:
- Data file paths
- Model hyperparameters
- Feature engineering parameters
- Output settings

## 📈 Performance Metrics

- **Cross-Validation Accuracy**: 99.33%
- **Model Confidence**: High (ensemble approach)
- **Generalization**: Good (balanced regularization)
- **Feature Importance**: Top features identified and ranked

## 🌍 Agricultural Applications

### Farmers
- Get crop recommendations based on soil tests
- Optimize nitrogen application timing
- Plan sustainable crop rotations
- Improve soil health and fertility

### Extension Workers
- Provide evidence-based recommendations
- Train farmers on best practices
- Monitor agricultural outcomes
- Support sustainable farming initiatives

### Researchers
- Analyze crop-environment relationships
- Study nitrogen cycling patterns
- Develop improved farming systems
- Validate agricultural practices

## 🚨 Important Notes

### Data Quality
- Input accurate soil test results
- Consider seasonal variations
- Monitor crop response to recommendations
- Adjust based on local conditions

### Model Limitations
- Trained on specific geographic regions
- May need local calibration
- Regular updates recommended
- Always validate with field trials

## 🔮 Future Enhancements

- [ ] Mobile application
- [ ] Weather integration
- [ ] Satellite imagery analysis
- [ ] Real-time soil sensors
- [ ] Advanced crop disease detection
- [ ] Market price integration

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Agricultural experts for domain knowledge
- Open-source ML community
- Farmers for feedback and validation
- Research institutions for data and insights

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in `docs/` folder

## 🖼️ Image Generation

The system automatically generates comprehensive visualizations for better understanding:

### Available Visualizations
- **Feature Importance Charts**: Shows which factors most influence crop recommendations
- **Crop Distribution Plots**: Displays crop frequency and distribution patterns
- **Nitrogen Analysis**: Comprehensive nitrogen management insights
- **Crop Rotation Diagrams**: Visual crop rotation recommendations
- **Performance Metrics**: Model accuracy and performance comparisons
- **Comprehensive Dashboard**: All insights in one view

### Generating Images
```bash
# Generate all visualizations
python src/auto_visualize.py

# Quick visualization only
python src/auto_visualize.py quick

# Individual visualization modules
python src/image_generator.py
python src/data_visualizer.py
```

### Image Output
All images are saved in `output/output_images/` with high resolution (300 DPI) for:
- Reports and presentations
- Web applications
- Print materials
- Documentation

## 🎉 Getting Started Checklist

- [ ] Install Python 3.8+
- [ ] Clone the repository
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run interactive model: `python src/interactive_crop_model.py`
- [ ] Generate visualizations: `python src/auto_visualize.py`
- [ ] Try sample predictions
- [ ] Explore output examples and images
- [ ] Read model performance summary
- [ ] Start using for your farming needs!

---

**🌾 Happy Farming with AI-Powered Recommendations! 🚀**

*Built with ❤️ for sustainable agriculture*
