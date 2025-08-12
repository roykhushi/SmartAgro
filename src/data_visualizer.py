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

class DataVisualizer:
    """Generate visualizations from actual data files"""
    
    def __init__(self, output_dir="output", images_dir="output/output_images"):
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data_files(self):
        """Load all available data files"""
        data_files = {}
        
        # Load CSV files
        csv_files = list(self.output_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                data_files[csv_file.stem] = pd.read_csv(csv_file)
                print(f"✅ Loaded: {csv_file.name}")
            except Exception as e:
                print(f"❌ Error loading {csv_file.name}: {e}")
        
        # Load text files for additional context
        txt_files = list(self.output_dir.glob("*.txt"))
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                data_files[f"{txt_file.stem}_content"] = content
                print(f"✅ Loaded: {txt_file.name}")
            except Exception as e:
                print(f"❌ Error loading {txt_file.name}: {e}")
        
        return data_files
    
    def create_crop_rotation_visualization(self, rotation_data):
        """Create crop rotation visualization from actual data"""
        if 'crop_rotation_examples' not in rotation_data:
            print("❌ Crop rotation data not found")
            return
        
        df = rotation_data['crop_rotation_examples']
        
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Crop rotation frequency
        if 'Current_Crop' in df.columns:
            crop_counts = df['Current_Crop'].value_counts()
            bars = ax1.bar(range(len(crop_counts)), crop_counts.values, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(crop_counts))))
            ax1.set_title('Current Crop Distribution', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Crop')
            ax1.set_ylabel('Count')
            ax1.set_xticks(range(len(crop_counts)))
            ax1.set_xticklabels(crop_counts.index, rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, crop_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Next crop recommendations
        if 'Next_Crop' in df.columns:
            next_crop_counts = df['Next_Crop'].value_counts()
            wedges, texts, autotexts = ax2.pie(next_crop_counts.values, 
                                              labels=next_crop_counts.index,
                                              autopct='%1.1f%%',
                                              colors=plt.cm.Set3(np.linspace(0, 1, len(next_crop_counts))),
                                              startangle=90)
            ax2.set_title('Next Crop Recommendations', fontweight='bold', fontsize=14)
            
            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'crop_rotation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Crop rotation analysis saved: {self.images_dir / 'crop_rotation_analysis.png'}")
    
    def create_nitrogen_analysis(self, nitrogen_data):
        """Create nitrogen analysis visualization from actual data"""
        if 'nitrogen_recommendations' not in nitrogen_data:
            print("❌ Nitrogen recommendations data not found")
            return
        
        df = nitrogen_data['nitrogen_recommendations']
        
        plt.figure(figsize=(16, 12))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Nitrogen levels by crop
        if 'Crop' in df.columns and 'Nitrogen' in df.columns:
            sns.boxplot(data=df, x='Crop', y='Nitrogen', ax=ax1)
            ax1.set_title('Nitrogen Levels by Crop', fontweight='bold', fontsize=14)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_ylabel('Nitrogen (kg/ha)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Nitrogen vs Phosphorus relationship
        if 'Nitrogen' in df.columns and 'Phosphorus' in df.columns:
            ax2.scatter(df['Nitrogen'], df['Phosphorus'], alpha=0.6, s=60)
            ax2.set_xlabel('Nitrogen (kg/ha)')
            ax2.set_ylabel('Phosphorus (kg/ha)')
            ax2.set_title('Nitrogen vs Phosphorus Relationship', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df['Nitrogen'], df['Phosphorus'], 1)
            p = np.poly1d(z)
            ax2.plot(df['Nitrogen'], p(df['Nitrogen']), "r--", alpha=0.8)
        
        # 3. Nitrogen status distribution
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(status_counts)))
            wedges, texts, autotexts = ax3.pie(status_counts.values, 
                                              labels=status_counts.index,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90)
            ax3.set_title('Nitrogen Status Distribution', fontweight='bold', fontsize=14)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 4. Action recommendations
        if 'Action' in df.columns:
            action_counts = df['Action'].value_counts().head(10)  # Top 10 actions
            bars = ax4.barh(range(len(action_counts)), action_counts.values, 
                          color=plt.cm.plasma(np.linspace(0, 1, len(action_counts))))
            ax4.set_title('Top Action Recommendations', fontweight='bold', fontsize=14)
            ax4.set_xlabel('Count')
            ax4.set_yticks(range(len(action_counts)))
            ax4.set_yticklabels(action_counts.index)
            
            # Add value labels
            for bar, count in zip(bars, action_counts.values):
                ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'nitrogen_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Nitrogen analysis saved: {self.images_dir / 'nitrogen_analysis.png'}")
    
    def create_sample_predictions_analysis(self, predictions_data):
        """Create analysis of sample predictions"""
        if 'sample_predictions' not in predictions_data:
            print("❌ Sample predictions data not found")
            return
        
        df = predictions_data['sample_predictions']
        
        plt.figure(figsize=(16, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Predicted crop distribution
        if 'Predicted_Crop' in df.columns:
            crop_counts = df['Predicted_Crop'].value_counts()
            bars = ax1.bar(range(len(crop_counts)), crop_counts.values, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(crop_counts))))
            ax1.set_title('Predicted Crop Distribution', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Crop')
            ax1.set_ylabel('Count')
            ax1.set_xticks(range(len(crop_counts)))
            ax1.set_xticklabels(crop_counts.index, rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, crop_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Temperature vs Rainfall scatter
        if 'Temperature' in df.columns and 'Rainfall' in df.columns:
            scatter = ax2.scatter(df['Temperature'], df['Rainfall'], 
                                c=df['Predicted_Crop'].astype('category').cat.codes if 'Predicted_Crop' in df.columns else 'blue',
                                alpha=0.6, s=50, cmap='tab10')
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Rainfall (mm)')
            ax2.set_title('Temperature vs Rainfall by Predicted Crop', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar if crops are available
            if 'Predicted_Crop' in df.columns:
                plt.colorbar(scatter, ax=ax2, label='Crop')
        
        # 3. pH distribution
        if 'pH_Value' in df.columns:
            ax3.hist(df['pH_Value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('pH Value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('pH Value Distribution', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # Add mean line
            mean_ph = df['pH_Value'].mean()
            ax3.axvline(mean_ph, color='red', linestyle='--', 
                       label=f'Mean: {mean_ph:.2f}')
            ax3.legend()
        
        # 4. Nutrient levels comparison
        nutrient_cols = [col for col in df.columns if col in ['Nitrogen', 'Phosphorus', 'Potassium']]
        if nutrient_cols:
            nutrient_data = df[nutrient_cols]
            nutrient_data.boxplot(ax=ax4)
            ax4.set_title('Nutrient Levels Distribution', fontweight='bold', fontsize=14)
            ax4.set_ylabel('Value (kg/ha)')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'sample_predictions_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Sample predictions analysis saved: {self.images_dir / 'sample_predictions_analysis.png'}")
    
    def create_comprehensive_dashboard(self, data_files):
        """Create a comprehensive dashboard combining all visualizations"""
        print("📊 Creating comprehensive dashboard...")
        
        # Create dashboard with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        fig.suptitle('Nitrogen Management System - Comprehensive Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Crop distribution (top left)
        if 'crop_rotation_examples' in data_files:
            df = data_files['crop_rotation_examples']
            if 'Current_Crop' in df.columns:
                ax1 = fig.add_subplot(gs[0, 0])
                crop_counts = df['Current_Crop'].value_counts()
                wedges, texts, autotexts = ax1.pie(crop_counts.values, 
                                                  labels=crop_counts.index,
                                                  autopct='%1.1f%%',
                                                  colors=plt.cm.Set3(np.linspace(0, 1, len(crop_counts))))
                ax1.set_title('Current Crop Distribution', fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
        
        # 2. Nitrogen levels (top right)
        if 'nitrogen_recommendations' in data_files:
            df = data_files['nitrogen_recommendations']
            if 'Nitrogen' in df.columns and 'Crop' in df.columns:
                ax2 = fig.add_subplot(gs[0, 1])
                sns.boxplot(data=df, x='Crop', y='Nitrogen', ax=ax2)
                ax2.set_title('Nitrogen Levels by Crop', fontweight='bold')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
                ax2.set_ylabel('Nitrogen (kg/ha)')
        
        # 3. Temperature vs Rainfall (middle left)
        if 'sample_predictions' in data_files:
            df = data_files['sample_predictions']
            if 'Temperature' in df.columns and 'Rainfall' in df.columns:
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.scatter(df['Temperature'], df['Rainfall'], alpha=0.6, s=40)
                ax3.set_xlabel('Temperature (°C)')
                ax3.set_ylabel('Rainfall (mm)')
                ax3.set_title('Temperature vs Rainfall', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        # 4. pH distribution (middle right)
        if 'sample_predictions' in data_files:
            df = data_files['sample_predictions']
            if 'pH_Value' in df.columns:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.hist(df['pH_Value'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                ax4.set_xlabel('pH Value')
                ax4.set_ylabel('Frequency')
                ax4.set_title('pH Value Distribution', fontweight='bold')
                ax4.grid(True, alpha=0.3)
        
        # 5. Nutrient comparison (bottom left)
        if 'sample_predictions' in data_files:
            df = data_files['sample_predictions']
            nutrient_cols = [col for col in df.columns if col in ['Nitrogen', 'Phosphorus', 'Potassium']]
            if nutrient_cols:
                ax5 = fig.add_subplot(gs[2, 0])
                nutrient_data = df[nutrient_cols]
                nutrient_data.boxplot(ax=ax5)
                ax5.set_title('Nutrient Levels', fontweight='bold')
                ax5.set_ylabel('Value (kg/ha)')
                ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
        
        # 6. Crop rotation flow (bottom right)
        if 'crop_rotation_examples' in data_files:
            df = data_files['crop_rotation_examples']
            if 'Current_Crop' in df.columns and 'Next_Crop' in df.columns:
                ax6 = fig.add_subplot(gs[2, 1])
                rotation_counts = df.groupby(['Current_Crop', 'Next_Crop']).size().unstack(fill_value=0)
                rotation_counts.plot(kind='bar', stacked=True, ax=ax6)
                ax6.set_title('Crop Rotation Flow', fontweight='bold')
                ax6.set_xlabel('Current Crop')
                ax6.set_ylabel('Count')
                ax6.legend(title='Next Crop', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
        
        # 7. Summary statistics (bottom center)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = "System Summary:\n"
        if 'crop_rotation_examples' in data_files:
            df = data_files['crop_rotation_examples']
            summary_text += f"• Total crop rotation examples: {len(df)}\n"
            if 'Current_Crop' in df.columns:
                summary_text += f"• Unique crops: {df['Current_Crop'].nunique()}\n"
        
        if 'nitrogen_recommendations' in data_files:
            df = data_files['nitrogen_recommendations']
            summary_text += f"• Nitrogen recommendations: {len(df)}\n"
            if 'Nitrogen' in df.columns:
                summary_text += f"• Average nitrogen level: {df['Nitrogen'].mean():.1f} kg/ha\n"
        
        if 'sample_predictions' in data_files:
            df = data_files['sample_predictions']
            summary_text += f"• Sample predictions: {len(df)}\n"
            if 'Temperature' in df.columns:
                summary_text += f"• Temperature range: {df['Temperature'].min():.1f}°C - {df['Temperature'].max():.1f}°C\n"
        
        ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes, 
                fontsize=12, verticalalignment='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'comprehensive_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Comprehensive dashboard saved: {self.images_dir / 'comprehensive_dashboard.png'}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations from available data"""
        print("🖼️  Starting data visualization generation...")
        
        # Load data files
        data_files = self.load_data_files()
        
        if not data_files:
            print("❌ No data files found to visualize")
            return
        
        print(f"📁 Found {len(data_files)} data files")
        
        # Generate specific visualizations
        self.create_crop_rotation_visualization(data_files)
        self.create_nitrogen_analysis(data_files)
        self.create_sample_predictions_analysis(data_files)
        
        # Create comprehensive dashboard
        self.create_comprehensive_dashboard(data_files)
        
        print("✅ All visualizations generated successfully!")
        print(f"📁 Images saved in: {self.images_dir}")

def main():
    """Generate visualizations from actual data"""
    visualizer = DataVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
