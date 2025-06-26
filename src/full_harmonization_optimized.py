#!/usr/bin/env python3
"""
Optimized Full Data Harmonization for CNN Flood Susceptibility
Process all 18 factors with optimized memory management
"""

import sys
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

class OptimizedFloodHarmonizer:
    def __init__(self):
        self.input_dir = Path("data/raw/Tahirpur")
        self.output_dir = Path("data/processed/harmonized_full")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # All 18 factors
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
        
        self.target_crs = CRS.from_string("EPSG:32645")
        self.target_resolution = 50  # Start with 50m resolution for speed
        
        print(f"🎯 Processing all {len(self.factors)} factors")
        print(f"📍 Target CRS: {self.target_crs}")
        print(f"📏 Target Resolution: {self.target_resolution}m")
    
    def get_optimal_bounds_and_transform(self):
        """Calculate optimal bounds and transform for all factors"""
        print("🗺️  Analyzing spatial extents of all factors...")
        
        all_bounds = []
        factor_info = {}
        
        for factor in self.factors:
            filepath = self.input_dir / factor
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        # Transform bounds to target CRS if needed
                        if src.crs != self.target_crs:
                            from rasterio.warp import transform_bounds
                            bounds = transform_bounds(src.crs, self.target_crs, *src.bounds)
                        else:
                            bounds = src.bounds
                        
                        all_bounds.append(bounds)
                        factor_info[factor] = {
                            'original_crs': src.crs,
                            'original_shape': src.shape,
                            'original_bounds': src.bounds,
                            'target_bounds': bounds
                        }
                        
                        print(f"✅ {factor:<35} {src.crs} → {src.shape}")
                        
                except Exception as e:
                    print(f"❌ Error with {factor}: {e}")
        
        if not all_bounds:
            raise ValueError("No valid factors found!")
        
        # Calculate union bounds (not intersection to avoid negative dimensions)
        min_x = min([bounds[0] for bounds in all_bounds])
        min_y = min([bounds[1] for bounds in all_bounds])
        max_x = max([bounds[2] for bounds in all_bounds])
        max_y = max([bounds[3] for bounds in all_bounds])
        
        target_bounds = (min_x, min_y, max_x, max_y)
        
        # Calculate target dimensions
        width = int((max_x - min_x) / self.target_resolution)
        height = int((max_y - min_y) / self.target_resolution)
        
        # Create transform
        target_transform = from_bounds(*target_bounds, width, height)
        
        print(f"\n📐 Target Grid Specifications:")
        print(f"   Bounds: {target_bounds}")
        print(f"   Dimensions: {height} x {width}")
        print(f"   Total pixels: {height * width:,}")
        print(f"   Estimated memory: {height * width * len(self.factors) * 4 / 1024 / 1024:.1f} MB")
        
        return target_bounds, (height, width), target_transform, factor_info
    
    def harmonize_single_factor(self, factor, target_shape, target_transform):
        """Harmonize a single factor with progress reporting"""
        input_path = self.input_dir / factor
        output_path = self.output_dir / factor
        
        if output_path.exists():
            print(f"⚡ {factor:<35} Already exists - skipping")
            return True
        
        if not input_path.exists():
            print(f"❌ {factor:<35} Missing from input")
            return False
        
        start_time = time.time()
        
        try:
            with rasterio.open(input_path) as src:
                height, width = target_shape
                
                # Create destination array with appropriate dtype
                dst_dtype = src.dtypes[0] if src.dtypes[0] != np.uint8 else np.float32
                dst_array = np.empty((height, width), dtype=dst_dtype)
                
                # Reproject with appropriate resampling method
                resampling_method = Resampling.nearest if src.dtypes[0] in [np.uint8, np.int16] else Resampling.bilinear
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=self.target_crs,
                    resampling=resampling_method,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata
                )
                
                # Prepare output profile
                profile = {
                    'driver': 'GTiff',
                    'height': height,
                    'width': width,
                    'count': 1,
                    'dtype': dst_dtype,
                    'crs': self.target_crs,
                    'transform': target_transform,
                    'nodata': src.nodata,
                    'compress': 'lzw',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256
                }
                
                # Write harmonized raster
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(dst_array, 1)
                
                # Calculate statistics
                processing_time = time.time() - start_time
                file_size = output_path.stat().st_size / (1024 * 1024)
                
                if src.nodata is not None:
                    valid_pixels = np.sum(dst_array != src.nodata)
                else:
                    valid_pixels = np.sum(~np.isnan(dst_array))
                
                total_pixels = height * width
                valid_percent = valid_pixels / total_pixels * 100
                
                print(f"✅ {factor:<35} "
                      f"{file_size:5.1f}MB, "
                      f"{valid_pixels:7,}/{total_pixels:7,} "
                      f"({valid_percent:5.1f}%), "
                      f"{processing_time:5.1f}s")
                
                return True
                
        except Exception as e:
            print(f"❌ {factor:<35} Failed: {e}")
            return False
    
    def run_full_harmonization(self):
        """Run complete harmonization for all factors"""
        print("🌊 FULL DATA HARMONIZATION - ALL 18 FACTORS")
        print("=" * 80)
        
        # Get target specifications
        target_bounds, target_shape, target_transform, factor_info = self.get_optimal_bounds_and_transform()
        
        print(f"\n🚀 PROCESSING ALL FACTORS")
        print("-" * 80)
        print(f"{'Factor':<35} {'Size':<8} {'Valid Pixels':<17} {'Coverage':<8} {'Time':<6}")
        print("-" * 80)
        
        # Process all factors
        successful = 0
        failed = 0
        total_start_time = time.time()
        
        for i, factor in enumerate(self.factors, 1):
            print(f"[{i:2d}/{len(self.factors)}] ", end="")
            
            if self.harmonize_single_factor(factor, target_shape, target_transform):
                successful += 1
            else:
                failed += 1
        
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 80)
        print(f"📊 HARMONIZATION COMPLETE")
        print("=" * 80)
        print(f"✅ Successfully processed: {successful}/{len(self.factors)} factors")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📁 Output directory: {self.output_dir}")
        
        if successful > 0:
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Validate harmonized data consistency")
            print(f"   2. Create factor stack for CNN training")
            print(f"   3. Train CNN with all {successful} factors")
            
            return True
        
        return False

if __name__ == "__main__":
    harmonizer = OptimizedFloodHarmonizer()
    success = harmonizer.run_full_harmonization()
    sys.exit(0 if success else 1)