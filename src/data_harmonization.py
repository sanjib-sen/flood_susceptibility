#!/usr/bin/env python3
"""
Data Harmonization for CNN Flood Susceptibility
Resolves CRS and resolution mismatches across all GIS factors
"""

import sys
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.mask

class FloodDataHarmonizer:
    """
    Harmonize all flood factors to consistent CRS, resolution and extent
    """
    
    def __init__(self, 
                 input_dir="data/raw/Tahirpur",
                 output_dir="data/processed/harmonized", 
                 target_resolution=30,
                 target_crs="EPSG:32645"):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_resolution = target_resolution
        self.target_crs = CRS.from_string(target_crs)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all factors
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
    
    def analyze_data_extents(self):
        """Analyze all raster extents to determine common area"""
        print("🗺️  ANALYZING DATA EXTENTS")
        print("=" * 50)
        
        all_bounds = []
        crs_info = {}
        
        for factor in self.factors:
            filepath = self.input_dir / factor
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        # Transform bounds to target CRS
                        if src.crs != self.target_crs:
                            from rasterio.warp import transform_bounds
                            bounds = transform_bounds(src.crs, self.target_crs, *src.bounds)
                        else:
                            bounds = src.bounds
                        
                        all_bounds.append(bounds)
                        crs_info[factor] = src.crs
                        
                        print(f"✅ {factor:<35} CRS: {src.crs}")
                        
                except Exception as e:
                    print(f"❌ Error analyzing {factor}: {e}")
        
        if not all_bounds:
            raise ValueError("No valid raster files found!")
        
        # Calculate common extent (intersection)
        # Use union instead of intersection to avoid negative dimensions
        min_x = min([bounds[0] for bounds in all_bounds])
        min_y = min([bounds[1] for bounds in all_bounds])
        max_x = max([bounds[2] for bounds in all_bounds])
        max_y = max([bounds[3] for bounds in all_bounds])
        
        common_bounds = (min_x, min_y, max_x, max_y)
        
        print(f"\n📏 COMMON EXTENT (in {self.target_crs}):")
        print(f"   West: {min_x:.6f}")
        print(f"   South: {min_y:.6f}")
        print(f"   East: {max_x:.6f}")
        print(f"   North: {max_y:.6f}")
        
        # Calculate target dimensions
        width = int((max_x - min_x) / self.target_resolution)
        height = int((max_y - min_y) / self.target_resolution)
        
        print(f"\n📐 TARGET GRID:")
        print(f"   Resolution: {self.target_resolution}m")
        print(f"   Dimensions: {height} x {width}")
        print(f"   Total pixels: {height * width:,}")
        
        return common_bounds, (height, width), crs_info
    
    def create_target_transform(self, bounds, shape):
        """Create target transform for reprojection"""
        height, width = shape
        transform = from_bounds(*bounds, width, height)
        return transform
    
    def harmonize_single_factor(self, factor, common_bounds, target_shape, target_transform):
        """Harmonize a single factor to target specifications"""
        input_path = self.input_dir / factor
        output_path = self.output_dir / factor
        
        if not input_path.exists():
            print(f"❌ Missing: {factor}")
            return False
        
        try:
            with rasterio.open(input_path) as src:
                # Prepare destination array
                height, width = target_shape
                dst_array = np.empty((height, width), dtype=src.dtypes[0])
                
                # Reproject data
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata
                )
                
                # Write harmonized raster
                profile = {
                    'driver': 'GTiff',
                    'height': height,
                    'width': width,
                    'count': 1,
                    'dtype': src.dtypes[0],
                    'crs': self.target_crs,
                    'transform': target_transform,
                    'nodata': src.nodata,
                    'compress': 'lzw'
                }
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(dst_array, 1)
                
                # Verify output
                file_size = output_path.stat().st_size / (1024 * 1024)
                valid_pixels = np.sum(~np.isnan(dst_array) if src.nodata is None 
                                    else dst_array != src.nodata)
                total_pixels = height * width
                
                print(f"✅ {factor:<35} "
                      f"Size: {file_size:.1f}MB, "
                      f"Valid: {valid_pixels:,}/{total_pixels:,} "
                      f"({valid_pixels/total_pixels*100:.1f}%)")
                
                return True
                
        except Exception as e:
            print(f"❌ Failed to harmonize {factor}: {e}")
            return False
    
    def harmonize_all_factors(self):
        """Harmonize all factors to consistent specifications"""
        print("🔧 HARMONIZING ALL FACTORS")
        print("=" * 50)
        
        # Analyze extents first
        common_bounds, target_shape, crs_info = self.analyze_data_extents()
        target_transform = self.create_target_transform(common_bounds, target_shape)
        
        print(f"\n🎯 HARMONIZATION PROGRESS")
        print("-" * 50)
        
        # Track progress
        successful = 0
        failed = 0
        start_time = time.time()
        
        for i, factor in enumerate(self.factors, 1):
            print(f"[{i:2d}/{len(self.factors)}] Processing {factor}...")
            
            if self.harmonize_single_factor(factor, common_bounds, target_shape, target_transform):
                successful += 1
            else:
                failed += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 HARMONIZATION SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully harmonized: {successful}/{len(self.factors)} factors")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Processing time: {elapsed_time:.1f} seconds")
        print(f"📁 Output directory: {self.output_dir}")
        
        return successful, failed
    
    def validate_harmonized_data(self):
        """Validate that all harmonized data has consistent properties"""
        print(f"\n🔍 VALIDATING HARMONIZED DATA")
        print("=" * 50)
        
        reference_info = None
        validation_results = []
        
        for factor in self.factors:
            filepath = self.output_dir / factor
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        info = {
                            'factor': factor,
                            'shape': src.shape,
                            'crs': src.crs,
                            'transform': src.transform,
                            'bounds': src.bounds
                        }
                        
                        if reference_info is None:
                            reference_info = info
                            print(f"📏 Reference: {factor}")
                            print(f"   Shape: {info['shape']}")
                            print(f"   CRS: {info['crs']}")
                            print(f"   Bounds: {[f'{b:.6f}' for b in info['bounds']]}")
                        
                        # Check consistency
                        consistent = (
                            info['shape'] == reference_info['shape'] and
                            info['crs'] == reference_info['crs'] and
                            np.allclose([info['transform'][i] for i in range(6)],
                                      [reference_info['transform'][i] for i in range(6)], rtol=1e-10)
                        )
                        
                        validation_results.append({
                            'factor': factor,
                            'consistent': consistent,
                            'info': info
                        })
                        
                        status = "✅" if consistent else "❌"
                        print(f"{status} {factor}")
                        
                except Exception as e:
                    print(f"❌ Error validating {factor}: {e}")
                    validation_results.append({
                        'factor': factor,
                        'consistent': False,
                        'error': str(e)
                    })
        
        consistent_count = sum(1 for r in validation_results if r['consistent'])
        total_count = len(validation_results)
        
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"   Consistent factors: {consistent_count}/{total_count}")
        
        if consistent_count == total_count:
            print("✅ All harmonized data is consistent!")
            print("🎯 Ready for CNN training pipeline")
        else:
            print("⚠️  Some inconsistencies detected - review required")
        
        return consistent_count == total_count
    
    def create_factor_stack(self, output_path="data/processed/factor_stack.npy"):
        """Create a numpy stack of all harmonized factors"""
        print(f"\n📦 CREATING FACTOR STACK")
        print("=" * 50)
        
        factor_arrays = []
        factor_names = []
        
        for factor in self.factors:
            filepath = self.output_dir / factor
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        data = src.read(1)
                        factor_arrays.append(data)
                        factor_names.append(factor)
                        print(f"✅ Loaded: {factor}")
                except Exception as e:
                    print(f"❌ Failed to load {factor}: {e}")
        
        if factor_arrays:
            # Stack all factors
            factor_stack = np.stack(factor_arrays, axis=-1)
            
            # Save stack
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path, factor_stack)
            
            # Save factor names
            names_path = output_path.parent / "factor_names.npy"
            np.save(names_path, factor_names)
            
            print(f"\n💾 FACTOR STACK SAVED")
            print(f"   Shape: {factor_stack.shape}")
            print(f"   File: {output_path}")
            print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            print(f"   Factor names: {names_path}")
            
            return factor_stack, factor_names
        
        return None, None

def main():
    """Main harmonization function"""
    print("🌊 CNN FLOOD SUSCEPTIBILITY - DATA HARMONIZATION")
    print("=" * 70)
    
    # Initialize harmonizer
    harmonizer = FloodDataHarmonizer()
    
    print(f"🎯 HARMONIZATION SETTINGS:")
    print(f"   Input directory: {harmonizer.input_dir}")
    print(f"   Output directory: {harmonizer.output_dir}")
    print(f"   Target CRS: {harmonizer.target_crs}")
    print(f"   Target resolution: {harmonizer.target_resolution}m")
    print(f"   Total factors: {len(harmonizer.factors)}")
    
    # Harmonize all factors
    successful, failed = harmonizer.harmonize_all_factors()
    
    if successful == 0:
        print("❌ No factors were successfully harmonized!")
        return False
    
    # Validate harmonized data
    if not harmonizer.validate_harmonized_data():
        print("⚠️  Validation failed - check harmonized data")
        return False
    
    # Create factor stack for CNN training
    factor_stack, factor_names = harmonizer.create_factor_stack()
    
    if factor_stack is not None:
        print(f"\n🎉 DATA HARMONIZATION COMPLETE!")
        print("=" * 70)
        print(f"✅ Harmonized factors: {successful}")
        print(f"✅ Data validation: Passed")
        print(f"✅ Factor stack: Created")
        print(f"✅ Ready for CNN training")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Run CNN training: uv run python src/flood_susceptibility_cnn.py")
        print(f"   2. Monitor training progress")
        print(f"   3. Evaluate model performance")
        print("=" * 70)
        
        return True
    else:
        print("❌ Failed to create factor stack")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)