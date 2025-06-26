#!/usr/bin/env python3
"""
CNN Data Preprocessing for Flood Susceptibility
Focus on data loading and preprocessing only
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

class FloodDataPreprocessor:
    """
    Simplified data preprocessor for flood susceptibility analysis
    """
    
    def __init__(self, data_dir="data/raw/Tahirpur", target_resolution=30, target_crs="EPSG:32645"):
        self.data_dir = Path(data_dir)
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        
        # Define factors - simplified list
        self.factors = [
            'Elevation.tif',
            'Slope.tif', 
            'Curvature.tif',
            'TWI.tif',
            'TPI.tif',
            'Rainfall.tif',
            'Distance_from_Waterbody.tif',
            'Drainage_density.tif',
            'NDVI_Taherpur_2025.tif',
            'NDWI_Taherpur_2025.tif',
            'NDBI_Taherpur_2025.tif',
            'NDMI_Taherpur_2025.tif',
            'BSI_Taherpur_2025.tif',
            'MSI_Taherpur_2025.tif',
            'SAVI_Taherpur_2025.tif',
            'WRI_Taherpur_2025.tif',
            'Lithology.tif',
            'Export_LULC_Taherpur_Sentinel2.tif'
        ]
    
    def load_raster_info(self, filepath):
        """Load basic raster information"""
        try:
            with rasterio.open(filepath) as src:
                info = {
                    'shape': src.shape,
                    'crs': src.crs,
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'dtype': src.dtypes[0],
                    'nodata': src.nodata
                }
                return info
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def analyze_data_characteristics(self):
        """Analyze all available raster files"""
        print("📊 ANALYZING DATA CHARACTERISTICS")
        print("=" * 50)
        
        available_factors = []
        all_bounds = []
        
        for factor in self.factors:
            filepath = self.data_dir / factor
            if filepath.exists():
                info = self.load_raster_info(filepath)
                if info:
                    available_factors.append((factor, info))
                    all_bounds.append(info['bounds'])
                    
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"✅ {factor:<35}")
                    print(f"   Shape: {info['shape']}")
                    print(f"   CRS: {info['crs']}")
                    print(f"   Size: {size_mb:.1f} MB")
                    print()
                else:
                    print(f"❌ Failed to load: {factor}")
            else:
                print(f"❌ Missing: {factor}")
        
        print(f"📈 SUMMARY:")
        print(f"   Available factors: {len(available_factors)}/{len(self.factors)}")
        
        if all_bounds:
            # Calculate common extent
            min_x = min([bounds[0] for bounds in all_bounds])
            min_y = min([bounds[1] for bounds in all_bounds])
            max_x = max([bounds[2] for bounds in all_bounds])
            max_y = max([bounds[3] for bounds in all_bounds])
            
            print(f"   Common extent: ({min_x:.6f}, {min_y:.6f}, {max_x:.6f}, {max_y:.6f})")
            
            # Estimate target dimensions
            width = int((max_x - min_x) / (self.target_resolution / 111000))  # Rough conversion
            height = int((max_y - min_y) / (self.target_resolution / 111000))
            
            print(f"   Estimated target dimensions: {height} x {width}")
            print(f"   Estimated memory requirement: {height * width * len(available_factors) * 4 / 1024 / 1024:.1f} MB")
        
        return available_factors
    
    def load_sample_data(self, max_factors=5):
        """Load a sample of data for testing"""
        print("\n🔬 LOADING SAMPLE DATA")
        print("=" * 50)
        
        available_factors = []
        for factor in self.factors[:max_factors]:  # Limit to first few factors
            filepath = self.data_dir / factor
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        # Read a small sample from the center
                        window = rasterio.windows.Window(
                            src.width//4, src.height//4,  # Start from 1/4 point
                            src.width//2, src.height//2   # Read 1/2 of each dimension
                        )
                        data = src.read(1, window=window)
                        
                        # Handle nodata
                        if src.nodata is not None:
                            data = np.where(data == src.nodata, np.nan, data)
                        
                        print(f"✅ {factor:<35} Sample shape: {data.shape}")
                        available_factors.append((factor, data))
                        
                except Exception as e:
                    print(f"❌ Failed to load sample from {factor}: {e}")
        
        if available_factors:
            print(f"\n📊 Successfully loaded {len(available_factors)} sample datasets")
            
            # Create a sample stack
            sample_stack = []
            min_shape = min([data.shape for _, data in available_factors])
            
            for factor, data in available_factors:
                # Crop to minimum shape
                cropped = data[:min_shape[0], :min_shape[1]]
                sample_stack.append(cropped)
            
            sample_stack = np.stack(sample_stack, axis=-1)
            print(f"   Sample stack shape: {sample_stack.shape}")
            
            # Calculate valid pixels
            valid_mask = ~np.isnan(sample_stack).any(axis=-1)
            valid_pixels = np.sum(valid_mask)
            total_pixels = sample_stack.shape[0] * sample_stack.shape[1]
            
            print(f"   Valid pixels: {valid_pixels:,}/{total_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
            
            return sample_stack, valid_mask
        
        return None, None
    
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs_to_create = [
            "data/processed",
            "outputs/figures", 
            "outputs/reports",
            "models/logs"
        ]
        
        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created/verified: {directory}")

def main():
    """Main preprocessing function"""
    print("🌊 CNN FLOOD SUSCEPTIBILITY - DATA PREPROCESSING")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = FloodDataPreprocessor()
    
    # Create output directories
    print("\n📁 SETTING UP OUTPUT DIRECTORIES")
    print("-" * 40)
    preprocessor.create_output_directories()
    
    # Analyze data characteristics
    available_factors = preprocessor.analyze_data_characteristics()
    
    if not available_factors:
        print("❌ No data available for processing!")
        return False
    
    # Load sample data for testing
    sample_stack, valid_mask = preprocessor.load_sample_data()
    
    if sample_stack is not None:
        # Save sample results
        print("\n💾 SAVING SAMPLE RESULTS")
        print("-" * 40)
        
        try:
            np.save("data/processed/sample_factor_stack.npy", sample_stack)
            np.save("data/processed/sample_valid_mask.npy", valid_mask)
            print("✅ Saved sample data to data/processed/")
        except Exception as e:
            print(f"❌ Failed to save sample data: {e}")
    
    # Generate preprocessing report
    print("\n📋 PREPROCESSING REPORT")
    print("=" * 70)
    print(f"✅ Environment setup: Complete")
    print(f"✅ Data availability: {len(available_factors)}/{len(preprocessor.factors)} factors")
    print(f"✅ Sample processing: {'Complete' if sample_stack is not None else 'Failed'}")
    print(f"✅ Output directories: Created")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Full data preprocessing: Ready")
    print(f"   2. CNN model training: Environment ready")
    print(f"   3. Model evaluation: Pending")
    
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)