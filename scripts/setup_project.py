#!/usr/bin/env python3
"""
Project Setup Script for CNN Flood Susceptibility Mapping
Creates necessary directories and validates project structure
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Create project directory structure if it doesn't exist"""
    
    # Define directory structure
    directories = [
        "src",
        "data/raw",
        "data/processed", 
        "data/interim",
        "data/external",
        "models/trained",
        "models/checkpoints",
        "models/logs",
        "outputs/figures",
        "outputs/maps", 
        "outputs/reports",
        "docs/papers",
        "docs/presentations",
        "docs/notes",
        "config",
        "scripts",
        "notebooks",
        "tests"
    ]
    
    print("🏗️  Setting up CNN Flood Susceptibility project structure...")
    
    created_dirs = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
            print(f"✅ Created: {directory}")
        else:
            print(f"✓ Exists: {directory}")
    
    if created_dirs:
        print(f"\n📁 Created {len(created_dirs)} new directories")
    else:
        print("\n📁 All directories already exist")

def validate_data_availability():
    """Check if required data files are available"""
    
    data_dir = Path("data/raw/Tahirpur")
    
    required_files = [
        # Topographic factors
        "Elevation.tif", "Slope.tif", "Curvature.tif", "TWI.tif", "TPI.tif",
        # Hydrological factors  
        "Rainfall.tif", "Distance_from_Waterbody.tif", "Drainage_density.tif",
        # Satellite indices
        "NDVI_Taherpur_2025.tif", "NDWI_Taherpur_2025.tif", "NDBI_Taherpur_2025.tif",
        "NDMI_Taherpur_2025.tif", "BSI_Taherpur_2025.tif", "MSI_Taherpur_2025.tif", 
        "WRI_Taherpur_2025.tif", "SAVI_Taherpur_2025.tif",
        # Categorical factors
        "Lithology.tif", "Export_LULC_Taherpur_Sentinel2.tif"
    ]
    
    print("\n📊 Validating data availability...")
    
    missing_files = []
    total_size = 0
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            total_size += file_size
            print(f"✅ {file:<35} ({file_size:.1f} MB)")
        else:
            missing_files.append(file)
            print(f"❌ {file:<35} (missing)")
    
    print(f"\n📈 Data Summary:")
    print(f"   Available: {len(required_files) - len(missing_files)}/{len(required_files)} files")
    print(f"   Total size: {total_size:.1f} MB")
    print(f"   Completeness: {(len(required_files) - len(missing_files))/len(required_files)*100:.1f}%")
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"     - {file}")
        return False
    else:
        print("\n✅ All required data files are available!")
        return True

def check_dependencies():
    """Check if required Python packages are available"""
    
    required_packages = [
        "numpy", "pandas", "rasterio", "tensorflow", 
        "scikit-learn", "matplotlib", "seaborn"
    ]
    
    print("\n🐍 Checking Python dependencies...")
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing {len(missing_packages)} packages:")
        for package in missing_packages:
            print(f"     - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r config/requirements.txt")
        return False
    else:
        print("\n✅ All required packages are available!")
        return True

def create_gitignore():
    """Create .gitignore file for the project"""
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.log
*.tmp
.cache/

# Processed data (stored in Git LFS)
data/processed/*.npy
data/processed/*.npz
data/interim/

# Model outputs
models/trained/*.h5
models/checkpoints/
models/logs/

# Temporary outputs
outputs/temp/
*.png
*.jpg
*.tif
"""

    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")
    else:
        print("✓ .gitignore file already exists")

def main():
    """Main setup function"""
    print("🌊 CNN Flood Susceptibility Mapping - Project Setup")
    print("=" * 55)
    
    # Create directory structure
    create_project_structure()
    
    # Validate data availability
    data_ok = validate_data_availability()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create gitignore
    create_gitignore()
    
    # Final status
    print("\n" + "=" * 55)
    print("🎯 Setup Summary:")
    print(f"   📁 Project structure: ✅ Complete")
    print(f"   📊 Data availability: {'✅ Ready' if data_ok else '⚠️  Incomplete'}")
    print(f"   🐍 Dependencies: {'✅ Ready' if deps_ok else '⚠️  Missing packages'}")
    
    if data_ok and deps_ok:
        print("\n🚀 Project is ready for CNN implementation!")
        print("\nNext steps:")
        print("   1. Run: python src/data_exploration.py")
        print("   2. Run: python src/flood_susceptibility_cnn.py")
    else:
        print("\n⚠️  Please resolve the issues above before proceeding.")
    
    print("=" * 55)

if __name__ == "__main__":
    main()