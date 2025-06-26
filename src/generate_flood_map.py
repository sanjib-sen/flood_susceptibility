#!/usr/bin/env python3
"""
Generate Flood Susceptibility Maps using Trained CNN
Creates final flood susceptibility maps for Tahirpur
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

class FloodMapGenerator:
    def __init__(self):
        self.harmonized_dir = Path("data/processed/harmonized")
        self.model_path = Path("models/trained/simple_flood_cnn.h5")
        self.output_dir = Path("outputs/maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Priority factors used in training
        self.factors = [
            'Elevation.tif',
            'Slope.tif',
            'Rainfall.tif',
            'NDVI_Taherpur_2025.tif',
            'NDWI_Taherpur_2025.tif'
        ]
        
        print(f"🗺️  Generating flood susceptibility maps using {len(self.factors)} factors")
    
    def load_trained_model(self):
        """Load the trained CNN model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")
        
        print(f"🧠 Loading trained model: {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully")
        
        return model
    
    def load_harmonized_factors(self):
        """Load all harmonized factors for prediction"""
        print("📊 Loading harmonized factors...")
        
        factor_arrays = []
        reference_profile = None
        
        for factor in self.factors:
            filepath = self.harmonized_dir / factor
            if filepath.exists():
                with rasterio.open(filepath) as src:
                    data = src.read(1)
                    factor_arrays.append(data)
                    
                    if reference_profile is None:
                        reference_profile = src.profile.copy()
                    
                    print(f"✅ Loaded {factor}: {data.shape}")
            else:
                raise FileNotFoundError(f"Missing harmonized factor: {factor}")
        
        # Stack factors
        factor_stack = np.stack(factor_arrays, axis=-1)
        print(f"📦 Complete factor stack: {factor_stack.shape}")
        
        return factor_stack, reference_profile
    
    def predict_flood_susceptibility(self, model, factor_stack):
        """Generate flood susceptibility predictions"""
        print("🔮 Generating flood susceptibility predictions...")
        
        # Get dimensions
        height, width, n_factors = factor_stack.shape
        
        # Prepare data for prediction
        # Reshape to (n_pixels, n_factors)
        factor_data = factor_stack.reshape(-1, n_factors)
        
        # Handle NaN values
        valid_mask = ~np.isnan(factor_data).any(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        print(f"📊 Valid pixels for prediction: {len(valid_indices):,}/{len(factor_data):,}")
        
        # Initialize prediction array
        predictions = np.full(len(factor_data), np.nan)
        
        if len(valid_indices) > 0:
            # Predict on valid data
            valid_data = factor_data[valid_indices]
            
            # Predict in batches to manage memory
            batch_size = 1000
            valid_predictions = []
            
            for i in range(0, len(valid_data), batch_size):
                batch = valid_data[i:i+batch_size]
                batch_pred = model.predict(batch, verbose=0)
                valid_predictions.extend(batch_pred.flatten())
            
            # Fill predictions
            predictions[valid_indices] = valid_predictions
        
        # Reshape back to spatial dimensions
        susceptibility_map = predictions.reshape(height, width)
        
        print(f"✅ Predictions complete")
        print(f"📈 Susceptibility range: {np.nanmin(susceptibility_map):.4f} - {np.nanmax(susceptibility_map):.4f}")
        
        return susceptibility_map, valid_mask.reshape(height, width)
    
    def classify_susceptibility(self, susceptibility_map):
        """Classify susceptibility into categories"""
        print("🏷️  Classifying susceptibility levels...")
        
        # Define thresholds
        valid_data = susceptibility_map[~np.isnan(susceptibility_map)]
        
        if len(valid_data) == 0:
            print("❌ No valid susceptibility data for classification")
            return np.full_like(susceptibility_map, 0)
        
        # Use percentiles for classification
        very_low = np.percentile(valid_data, 20)
        low = np.percentile(valid_data, 40)
        moderate = np.percentile(valid_data, 60)
        high = np.percentile(valid_data, 80)
        
        # Classify
        classified = np.full_like(susceptibility_map, 0, dtype=int)
        classified[susceptibility_map >= very_low] = 1  # Very Low
        classified[susceptibility_map >= low] = 2       # Low
        classified[susceptibility_map >= moderate] = 3  # Moderate
        classified[susceptibility_map >= high] = 4      # High
        classified[susceptibility_map >= np.percentile(valid_data, 80)] = 5  # Very High
        
        # Handle NaN values
        classified[np.isnan(susceptibility_map)] = 0
        
        print(f"📊 Classification distribution:")
        for i, label in enumerate(['No Data', 'Very Low', 'Low', 'Moderate', 'High', 'Very High']):
            count = np.sum(classified == i)
            percentage = count / classified.size * 100
            print(f"   {label}: {count:,} pixels ({percentage:.1f}%)")
        
        return classified
    
    def save_susceptibility_map(self, susceptibility_map, reference_profile, suffix="continuous"):
        """Save susceptibility map as GeoTIFF"""
        output_path = self.output_dir / f"flood_susceptibility_{suffix}.tif"
        
        # Update profile
        profile = reference_profile.copy()
        profile.update({
            'dtype': np.float32,
            'nodata': np.nan,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(susceptibility_map.astype(np.float32), 1)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"💾 Saved {suffix} map: {output_path} ({file_size:.1f} MB)")
        
        return output_path
    
    def save_classified_map(self, classified_map, reference_profile):
        """Save classified susceptibility map"""
        output_path = self.output_dir / "flood_susceptibility_classified.tif"
        
        # Update profile for integer data
        profile = reference_profile.copy()
        profile.update({
            'dtype': np.uint8,
            'nodata': 0,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classified_map.astype(np.uint8), 1)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"💾 Saved classified map: {output_path} ({file_size:.1f} MB)")
        
        return output_path
    
    def create_visualization(self, susceptibility_map, classified_map):
        """Create visualization of flood susceptibility"""
        print("📊 Creating flood susceptibility visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Continuous susceptibility map
        im1 = ax1.imshow(susceptibility_map, cmap='RdYlBu_r', interpolation='nearest')
        ax1.set_title('Flood Susceptibility\n(Continuous Values)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Longitude (Grid Cells)')
        ax1.set_ylabel('Latitude (Grid Cells)')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Susceptibility Score', rotation=270, labelpad=15)
        
        # Classified susceptibility map
        class_colors = ['#ffffff', '#2166ac', '#5aae61', '#fee08b', '#f46d43', '#a50026']
        class_cmap = colors.ListedColormap(class_colors)
        
        im2 = ax2.imshow(classified_map, cmap=class_cmap, vmin=0, vmax=5, interpolation='nearest')
        ax2.set_title('Flood Susceptibility\n(Classified)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude (Grid Cells)')
        ax2.set_ylabel('Latitude (Grid Cells)')
        
        # Add legend for classified map
        class_labels = ['No Data', 'Very Low', 'Low', 'Moderate', 'High', 'Very High']
        legend_elements = [Rectangle((0,0),1,1, facecolor=class_colors[i], 
                                   label=class_labels[i]) for i in range(len(class_labels))]
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / "flood_susceptibility_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualization saved: {viz_path}")
        
        return viz_path
    
    def generate_summary_report(self, susceptibility_map, classified_map):
        """Generate summary report"""
        report_path = self.output_dir / "flood_susceptibility_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("FLOOD SUSCEPTIBILITY MAPPING REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Study Area: Tahirpur, Bangladesh\n")
            f.write(f"CNN Model: Simple Feed-Forward Network\n")
            f.write(f"Factors Used: {len(self.factors)}\n")
            f.write(f"Grid Resolution: ~100m\n")
            f.write(f"Total Area: {susceptibility_map.shape[0]} x {susceptibility_map.shape[1]} pixels\n\n")
            
            f.write("FACTOR LIST:\n")
            f.write("-" * 20 + "\n")
            for i, factor in enumerate(self.factors, 1):
                f.write(f"{i}. {factor}\n")
            
            f.write(f"\nSUSCEPTIBILITY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            valid_data = susceptibility_map[~np.isnan(susceptibility_map)]
            if len(valid_data) > 0:
                f.write(f"Mean Susceptibility: {np.mean(valid_data):.4f}\n")
                f.write(f"Std Deviation: {np.std(valid_data):.4f}\n")
                f.write(f"Minimum: {np.min(valid_data):.4f}\n")
                f.write(f"Maximum: {np.max(valid_data):.4f}\n")
            
            f.write(f"\nCLASSIFICATION SUMMARY:\n")
            f.write("-" * 25 + "\n")
            class_labels = ['No Data', 'Very Low', 'Low', 'Moderate', 'High', 'Very High']
            for i, label in enumerate(class_labels):
                count = np.sum(classified_map == i)
                percentage = count / classified_map.size * 100
                f.write(f"{label}: {count:,} pixels ({percentage:.1f}%)\n")
            
            f.write(f"\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("1. flood_susceptibility_continuous.tif\n")
            f.write("2. flood_susceptibility_classified.tif\n")
            f.write("3. flood_susceptibility_visualization.png\n")
            f.write("4. flood_susceptibility_report.txt\n")
        
        print(f"📋 Summary report saved: {report_path}")
        
        return report_path
    
    def run_mapping(self):
        """Run complete flood susceptibility mapping"""
        print("🌊 FLOOD SUSCEPTIBILITY MAPPING")
        print("=" * 60)
        
        try:
            # Load trained model
            model = self.load_trained_model()
            
            # Load harmonized factors
            factor_stack, reference_profile = self.load_harmonized_factors()
            
            # Generate predictions
            susceptibility_map, valid_mask = self.predict_flood_susceptibility(model, factor_stack)
            
            # Classify susceptibility
            classified_map = self.classify_susceptibility(susceptibility_map)
            
            # Save maps
            continuous_path = self.save_susceptibility_map(susceptibility_map, reference_profile, "continuous")
            classified_path = self.save_classified_map(classified_map, reference_profile)
            
            # Create visualization
            viz_path = self.create_visualization(susceptibility_map, classified_map)
            
            # Generate report
            report_path = self.generate_summary_report(susceptibility_map, classified_map)
            
            print("\n🎉 FLOOD MAPPING COMPLETE!")
            print("=" * 60)
            print("✅ Continuous susceptibility map generated")
            print("✅ Classified susceptibility map generated")
            print("✅ Visualization created")
            print("✅ Summary report generated")
            
            print(f"\n📁 Output files in: {self.output_dir}")
            print("🎯 Ready for analysis and interpretation")
            
            return True
            
        except Exception as e:
            print(f"❌ Mapping failed: {e}")
            return False

if __name__ == "__main__":
    generator = FloodMapGenerator()
    success = generator.run_mapping()
    sys.exit(0 if success else 1)