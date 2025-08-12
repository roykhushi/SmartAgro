import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class ImageGenerator:
    """Generate visualizations for the nitrogen management system"""
    
    def __init__(self, output_dir="output/output_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_feature_importance_plot(self, feature_importance_data, title="Feature Importance"):
        """Create feature importance visualization"""
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_data.items(), 
                               key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(features)), importance, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importance)):
            plt.text(imp + 0.01, i, f'{imp:.3f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved: {self.output_dir / 'feature_importance.png'}")
    
    def create_crop_distribution_plot(self, crop_data, title="Crop Distribution"):
        """Create crop distribution visualization"""
        plt.figure(figsize=(12, 8))
        
        # Count crop occurrences
        crop_counts = crop_data.value_counts()
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
        wedges, texts, autotexts = plt.pie(crop_counts.values, 
                                          labels=crop_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crop_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Crop distribution plot saved: {self.output_dir / 'crop_distribution.png'}")
    
    def create_nitrogen_recommendations_chart(self, nitrogen_data, title="Nitrogen Recommendations"):
        """Create nitrogen recommendations visualization"""
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Nitrogen levels by crop
        if 'Crop' in nitrogen_data.columns and 'Nitrogen' in nitrogen_data.columns:
            sns.boxplot(data=nitrogen_data, x='Crop', y='Nitrogen', ax=ax1)
            ax1.set_title('Nitrogen Levels by Crop', fontweight='bold')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_ylabel('Nitrogen (kg/ha)')
        
        # 2. Nitrogen vs Phosphorus scatter
        if 'Nitrogen' in nitrogen_data.columns and 'Phosphorus' in nitrogen_data.columns:
            ax2.scatter(nitrogen_data['Nitrogen'], nitrogen_data['Phosphorus'], 
                       alpha=0.6, s=50)
            ax2.set_xlabel('Nitrogen (kg/ha)')
            ax2.set_ylabel('Phosphorus (kg/ha)')
            ax2.set_title('Nitrogen vs Phosphorus Relationship', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Optimal nitrogen ranges
        optimal_ranges = {
            'Rice': (80, 120),
            'Wheat': (60, 100),
            'Maize': (70, 110),
            'Cotton': (60, 100),
            'Sugarcane': (80, 120)
        }
        
        crops = list(optimal_ranges.keys())
        min_n = [optimal_ranges[crop][0] for crop in crops]
        max_n = [optimal_ranges[crop][1] for crop in crops]
        
        x_pos = np.arange(len(crops))
        ax3.bar(x_pos, max_n, alpha=0.3, label='Maximum', color='lightcoral')
        ax3.bar(x_pos, min_n, alpha=0.7, label='Minimum', color='lightblue')
        ax3.set_xlabel('Crop')
        ax3.set_ylabel('Nitrogen (kg/ha)')
        ax3.set_title('Optimal Nitrogen Ranges by Crop', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(crops, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Nitrogen application timing
        timing_data = {
            'Crop': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'],
            'Sowing': [50, 50, 30, 25, 40],
            'Vegetative': [25, 25, 40, 50, 30],
            'Reproductive': [25, 25, 30, 25, 30]
        }
        
        timing_df = pd.DataFrame(timing_data)
        timing_df.set_index('Crop').plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Nitrogen Application Timing (%)', fontweight='bold')
        ax4.set_ylabel('Percentage')
        ax4.set_xlabel('Crop')
        ax4.legend(title='Growth Stage')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'nitrogen_recommendations.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Nitrogen recommendations chart saved: {self.output_dir / 'nitrogen_recommendations.png'}")
    
    def create_crop_rotation_diagram(self, rotation_data, title="Crop Rotation Suggestions"):
        """Create crop rotation visualization"""
        plt.figure(figsize=(14, 10))
        
        # Create a circular diagram
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Define crops and their positions
        crops = rotation_data['Crop'].unique() if 'Crop' in rotation_data.columns else ['Rice', 'Wheat', 'Maize', 'Cotton', 'Pulses']
        n_crops = len(crops)
        
        # Calculate angles
        angles = np.linspace(0, 2*np.pi, n_crops, endpoint=False)
        
        # Plot crop positions
        ax.scatter(angles, [1]*n_crops, s=200, c='lightblue', edgecolors='darkblue', linewidth=2)
        
        # Add crop labels
        for i, (angle, crop) in enumerate(zip(angles, crops)):
            ax.text(angle, 1.2, crop, ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Add rotation arrows
        for i in range(n_crops):
            start_angle = angles[i]
            end_angle = angles[(i+1) % n_crops]
            
            # Create curved arrow
            arrow_angles = np.linspace(start_angle, end_angle, 20)
            arrow_radius = 0.8
            ax.plot(arrow_angles, [arrow_radius]*20, 'k-', linewidth=2, alpha=0.7)
            
            # Add arrowhead
            mid_angle = (start_angle + end_angle) / 2
            ax.arrow(mid_angle, arrow_radius, 0.1, 0, 
                    head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.7)
        
        ax.set_ylim(0, 1.5)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crop_rotation_diagram.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Crop rotation diagram saved: {self.output_dir / 'crop_rotation_diagram.png'}")
    
    def create_performance_metrics_chart(self, performance_data, title="Model Performance Metrics"):
        """Create model performance visualization"""
        plt.figure(figsize=(16, 10))
        
        # Create subplots for different metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        if 'Model' in performance_data.columns and 'Accuracy' in performance_data.columns:
            bars = ax1.bar(performance_data['Model'], performance_data['Accuracy'], 
                          color=plt.cm.viridis(np.linspace(0, 1, len(performance_data))))
            ax1.set_title('Model Accuracy Comparison', fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for bar, acc in zip(bars, performance_data['Accuracy']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precision vs Recall
        if 'Precision' in performance_data.columns and 'Recall' in performance_data.columns:
            ax2.scatter(performance_data['Precision'], performance_data['Recall'], 
                       s=100, alpha=0.7, c=range(len(performance_data)), cmap='viridis')
            ax2.set_xlabel('Precision')
            ax2.set_ylabel('Recall')
            ax2.set_title('Precision vs Recall', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add model labels
            for i, model in enumerate(performance_data['Model']):
                ax2.annotate(model, (performance_data['Precision'].iloc[i], 
                                   performance_data['Recall'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. F1 Score comparison
        if 'F1_Score' in performance_data.columns:
            bars = ax3.bar(performance_data['Model'], performance_data['F1_Score'], 
                          color=plt.cm.plasma(np.linspace(0, 1, len(performance_data))))
            ax3.set_title('F1 Score Comparison', fontweight='bold')
            ax3.set_ylabel('F1 Score')
            ax3.set_ylim(0, 1)
            
            # Add value labels
            for bar, f1 in zip(bars, performance_data['F1_Score']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Training time comparison
        if 'Training_Time' in performance_data.columns:
            bars = ax4.bar(performance_data['Model'], performance_data['Training_Time'], 
                          color=plt.cm.coolwarm(np.linspace(0, 1, len(performance_data))))
            ax4.set_title('Training Time Comparison', fontweight='bold')
            ax4.set_ylabel('Training Time (seconds)')
            
            # Add value labels
            for bar, time in zip(bars, performance_data['Training_Time']):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance metrics chart saved: {self.output_dir / 'performance_metrics.png'}")
    
    def create_sample_data_visualizations(self):
        """Create sample visualizations using available data"""
        print("🖼️  Generating sample visualizations...")
        
        # Sample feature importance data
        feature_importance = {
            'Nitrogen': 0.25,
            'Phosphorus': 0.20,
            'Potassium': 0.18,
            'Temperature': 0.15,
            'Humidity': 0.12,
            'pH_Value': 0.08,
            'Rainfall': 0.02
        }
        self.create_feature_importance_plot(feature_importance)
        
        # Sample crop distribution data
        crop_data = pd.Series(['Rice', 'Wheat', 'Maize', 'Cotton', 'Pulses'] * 20)
        self.create_crop_distribution_plot(crop_data)
        
        # Sample nitrogen recommendations data
        nitrogen_data = pd.DataFrame({
            'Crop': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'] * 4,
            'Nitrogen': np.random.uniform(50, 120, 20),
            'Phosphorus': np.random.uniform(30, 80, 20)
        })
        self.create_nitrogen_recommendations_chart(nitrogen_data)
        
        # Sample crop rotation data
        rotation_data = pd.DataFrame({
            'Crop': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Pulses'],
            'Next_Crop': ['Wheat', 'Pulses', 'Wheat', 'Wheat', 'Wheat']
        })
        self.create_crop_rotation_diagram(rotation_data)
        
        # Sample performance data
        performance_data = pd.DataFrame({
            'Model': ['Random Forest', 'SVM', 'Neural Network', 'XGBoost'],
            'Accuracy': [0.92, 0.89, 0.91, 0.93],
            'Precision': [0.91, 0.88, 0.90, 0.92],
            'Recall': [0.90, 0.87, 0.89, 0.91],
            'F1_Score': [0.90, 0.87, 0.89, 0.91],
            'Training_Time': [2.5, 1.8, 15.2, 3.1]
        })
        self.create_performance_metrics_chart(performance_data)
        
        print("✅ All sample visualizations generated successfully!")

def main():
    """Generate sample visualizations"""
    generator = ImageGenerator()
    generator.create_sample_data_visualizations()

if __name__ == "__main__":
    main()
