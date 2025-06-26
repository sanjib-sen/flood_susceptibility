#!/usr/bin/env python3
"""
Quick Data Harmonization - Simplified approach
Focus on essential factors and faster processing
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

class QuickHarmonizer:
    def __init__(self):
        self.input_dir = Path("data/raw/Tahirpur")
        self.output_dir = Path("data/processed/harmonized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use most reliable factors first
        self.priority_factors = [
            'Elevation.tif',
            'Slope.tif',
            'Rainfall.tif',
            'NDVI_Taherpur_2025.tif',
            'NDWI_Taherpur_2025.tif'
        ]
        
        self.target_crs = CRS.from_string("EPSG:32645")
        self.target_resolution = 30
    
    def get_reference_grid(self):
        """Use elevation as reference for target grid"""
        elevation_path = self.input_dir / "Elevation.tif"
        
        with rasterio.open(elevation_path) as src:
            # Use elevation bounds and reduce resolution to 100m for faster processing
            bounds = src.bounds
            
            # Calculate dimensions for 100m resolution
            width = int((bounds[2] - bounds[0]) / 100)
            height = int((bounds[3] - bounds[1]) / 100)
            
            from rasterio.transform import from_bounds
            transform = from_bounds(*bounds, width, height)
            
            print(f"📏 Reference Grid:")
            print(f"   Bounds: {bounds}")
            print(f"   Dimensions: {height} x {width}")
            print(f"   Resolution: ~100m")
            
            return bounds, (height, width), transform
    
    def harmonize_factor(self, factor, target_bounds, target_shape, target_transform):
        """Harmonize single factor"""
        input_path = self.input_dir / factor
        output_path = self.output_dir / factor
        
        try:
            with rasterio.open(input_path) as src:
                height, width = target_shape
                
                # Create destination array
                dst_array = np.empty((height, width), dtype=np.float32)
                
                # Reproject
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear
                )
                
                # Write output
                profile = {
                    'driver': 'GTiff',
                    'height': height,
                    'width': width,
                    'count': 1,
                    'dtype': np.float32,
                    'crs': self.target_crs,
                    'transform': target_transform,
                    'compress': 'lzw'
                }
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(dst_array, 1)
                
                print(f"✅ {factor}")
                return True
                
        except Exception as e:
            print(f"❌ {factor}: {e}")
            return False
    
    def run_quick_harmonization(self):
        """Run quick harmonization on priority factors"""
        print("🚀 QUICK HARMONIZATION")
        print("=" * 50)
        
        # Get reference grid
        target_bounds, target_shape, target_transform = self.get_reference_grid()
        
        # Process priority factors
        successful = 0
        for factor in self.priority_factors:
            if self.harmonize_factor(factor, target_bounds, target_shape, target_transform):
                successful += 1
        
        print(f"\n📊 Results: {successful}/{len(self.priority_factors)} successful")
        
        if successful > 0:
            print("✅ Quick harmonization complete!")
            print("🎯 Ready for initial CNN testing")
            return True
        
        return False

if __name__ == "__main__":
    harmonizer = QuickHarmonizer()
    success = harmonizer.run_quick_harmonization()
    sys.exit(0 if success else 1)