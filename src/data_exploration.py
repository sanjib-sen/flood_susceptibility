#!/usr/bin/env python3
"""
Data Exploration for Flood Susceptibility Analysis
Simple exploration using basic Python libraries
"""

import os
import sys
import subprocess
from pathlib import Path

def check_gdal_installation():
    """Check if GDAL is available"""
    try:
        result = subprocess.run(['gdalinfo', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ GDAL available: {result.stdout.strip()}")
            return True
        else:
            print("✗ GDAL not available")
            return False
    except FileNotFoundError:
        print("✗ GDAL not found in PATH")
        return False

def explore_raster_with_gdal(filepath):
    """Explore raster file using GDAL command line tools"""
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return None
    
    try:
        # Get basic info
        result = subprocess.run(['gdalinfo', str(filepath)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            info = result.stdout
            print(f"\n{'='*60}")
            print(f"FILE: {os.path.basename(filepath)}")
            print(f"{'='*60}")
            
            # Extract key information
            lines = info.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in 
                      ['size is', 'pixel size', 'origin', 'coordinate system', 
                       'band 1', 'minimum=', 'maximum=', 'nodata']):
                    print(line.strip())
            
            return info
        else:
            print(f"✗ Error reading {filepath}: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("✗ gdalinfo command not found")
        return None

def explore_data_directory(data_dir):
    """Explore all raster files in the data directory"""
    print("=== TAHIRPUR FLOOD SUSCEPTIBILITY DATA EXPLORATION ===\n")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return
    
    # Check GDAL availability
    gdal_available = check_gdal_installation()
    
    if not gdal_available:
        print("\nInstalling GDAL tools...")
        try:
            subprocess.run(['sudo', 'apt', 'update'], check=True, capture_output=True)
            subprocess.run(['sudo', 'apt', 'install', '-y', 'gdal-bin'], 
                          check=True, capture_output=True)
            print("✓ GDAL installed successfully")
            gdal_available = True
        except subprocess.CalledProcessError:
            print("✗ Failed to install GDAL")
            gdal_available = False
    
    # Find all TIF files
    tif_files = list(data_path.glob("*.tif"))
    
    if not tif_files:
        print(f"No .tif files found in {data_dir}")
        return
    
    print(f"\nFound {len(tif_files)} raster files:")
    for f in tif_files:
        print(f"  - {f.name}")
    
    # Categorize files
    topographic_factors = [
        'Elevation.tif', 'Slope.tif', 'Curvature.tif', 'TWI.tif', 'TPI.tif'
    ]
    
    hydrological_factors = [
        'Rainfall.tif', 'Distance_from_Waterbody.tif', 'Drainage_density.tif'
    ]
    
    satellite_indices = [
        'NDVI_Taherpur_2025.tif', 'NDWI_Taherpur_2025.tif', 
        'NDBI_Taherpur_2025.tif', 'NDMI_Taherpur_2025.tif',
        'BSI_Taherpur_2025.tif', 'MSI_Taherpur_2025.tif',
        'WRI_Taherpur_2025.tif', 'SAVI_Taherpur_2025.tif'
    ]
    
    categorical_factors = [
        'Lithology.tif', 'Export_LULC_Taherpur_Sentinel2.tif'
    ]
    
    # Explore each category
    categories = {
        "TOPOGRAPHIC FACTORS": topographic_factors,
        "HYDROLOGICAL FACTORS": hydrological_factors, 
        "SATELLITE INDICES": satellite_indices,
        "CATEGORICAL FACTORS": categorical_factors
    }
    
    for category, factor_list in categories.items():
        print(f"\n\n{'='*80}")
        print(f"{category}")
        print('='*80)
        
        available_factors = []
        for factor in factor_list:
            factor_path = data_path / factor
            if factor_path.exists():
                available_factors.append(factor)
                if gdal_available:
                    explore_raster_with_gdal(factor_path)
            else:
                print(f"✗ Missing: {factor}")
        
        print(f"\nAvailable {category.lower()}: {len(available_factors)}/{len(factor_list)}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    total_available = len(tif_files)
    total_expected = (len(topographic_factors) + len(hydrological_factors) + 
                     len(satellite_indices) + len(categorical_factors))
    
    print(f"Total raster files available: {total_available}")
    print(f"Total expected factors: {total_expected}")
    print(f"Data completeness: {total_available/total_expected*100:.1f}%")
    
    if gdal_available:
        print("\n✓ Ready for CNN preprocessing")
        print("✓ GDAL tools available for raster processing")
    else:
        print("\n⚠ GDAL tools needed for raster processing")
    
    print("\nNext steps:")
    print("1. Install required Python packages (rasterio, tensorflow, etc.)")
    print("2. Run data preprocessing pipeline")
    print("3. Train CNN model")
    print("4. Generate flood susceptibility maps")

def main():
    """Main execution"""
    data_directory = "data/raw/Tahirpur"
    explore_data_directory(data_directory)

if __name__ == "__main__":
    main()