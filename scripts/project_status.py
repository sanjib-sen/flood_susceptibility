#!/usr/bin/env python3
"""
Project Status Overview for CNN Flood Susceptibility Mapping
Provides comprehensive status of all project components
"""

import os
import sys
from pathlib import Path
import subprocess

def check_project_structure():
    """Check if project structure is complete"""
    
    required_dirs = [
        "src", "data/raw/Tahirpur", "data/processed", "models/trained", 
        "outputs/figures", "docs/papers", "config"
    ]
    
    print("📁 PROJECT STRUCTURE STATUS")
    print("-" * 40)
    
    all_exist = True
    for directory in required_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory}")
            all_exist = False
    
    return all_exist

def check_source_code():
    """Check status of source code files"""
    
    src_files = [
        "src/flood_susceptibility_cnn.py",
        "src/data_exploration.py", 
        "src/simple_cnn_approach.py"
    ]
    
    print("\n💻 SOURCE CODE STATUS")
    print("-" * 40)
    
    for file in src_files:
        file_path = Path(file)
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"✅ {file:<35} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {file:<35} (missing)")

def check_data_status():
    """Check data availability and processing status"""
    
    print("\n📊 DATA STATUS")
    print("-" * 40)
    
    # Raw data
    raw_data_dir = Path("data/raw/Tahirpur")
    if raw_data_dir.exists():
        tif_files = list(raw_data_dir.glob("*.tif"))
        total_size = sum(f.stat().st_size for f in tif_files) / (1024 * 1024)
        print(f"✅ Raw data: {len(tif_files)} TIF files ({total_size:.1f} MB)")
    else:
        print("❌ Raw data directory missing")
    
    # Processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        npy_files = list(processed_dir.glob("*.npy"))
        if npy_files:
            processed_size = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)
            print(f"✅ Processed data: {len(npy_files)} files ({processed_size:.1f} MB)")
        else:
            print("⏳ Processed data: Not yet created")
    else:
        print("❌ Processed data directory missing")

def check_model_status():
    """Check model training and output status"""
    
    print("\n🧠 MODEL STATUS")
    print("-" * 40)
    
    # Check for trained models
    models_dir = Path("models/trained")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.h5"))
        if model_files:
            model_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
            print(f"✅ Trained models: {len(model_files)} files ({model_size:.1f} MB)")
        else:
            print("⏳ Trained models: Not yet created")
    else:
        print("❌ Models directory missing")
    
    # Check for checkpoints
    checkpoints_dir = Path("models/checkpoints")
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*"))
        if checkpoint_files:
            print(f"✅ Checkpoints: {len(checkpoint_files)} files")
        else:
            print("⏳ Checkpoints: None yet")
    else:
        print("❌ Checkpoints directory missing")

def check_outputs():
    """Check generated outputs and results"""
    
    print("\n📈 OUTPUTS STATUS")
    print("-" * 40)
    
    # Check figures
    figures_dir = Path("outputs/figures")
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*"))
        if figure_files:
            print(f"✅ Figures: {len(figure_files)} files")
        else:
            print("⏳ Figures: Not yet generated")
    else:
        print("❌ Figures directory missing")
    
    # Check maps
    maps_dir = Path("outputs/maps")
    if maps_dir.exists():
        map_files = list(maps_dir.glob("*.tif"))
        if map_files:
            print(f"✅ Susceptibility maps: {len(map_files)} files")
        else:
            print("⏳ Susceptibility maps: Not yet generated")
    else:
        print("❌ Maps directory missing")

def check_dependencies():
    """Quick dependency check"""
    
    print("\n🐍 DEPENDENCIES STATUS")
    print("-" * 40)
    
    key_packages = ["numpy", "rasterio", "tensorflow", "scikit-learn"]
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (not installed)")

def get_git_status():
    """Get git repository status if available"""
    
    print("\n📝 GIT STATUS")
    print("-" * 40)
    
    try:
        # Check if in git repo
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            changes = result.stdout.strip().split('\n') if result.stdout.strip() else []
            if changes and changes[0]:  # Filter out empty strings
                print(f"⚠️  Uncommitted changes: {len(changes)} files")
                for change in changes[:5]:  # Show first 5
                    print(f"   {change}")
                if len(changes) > 5:
                    print(f"   ... and {len(changes) - 5} more")
            else:
                print("✅ Working directory clean")
                
            # Check branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                         capture_output=True, text=True, timeout=5)
            if branch_result.returncode == 0:
                branch = branch_result.stdout.strip()
                print(f"📍 Current branch: {branch}")
        else:
            print("❌ Not a git repository or git not available")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Git not available or timeout")

def calculate_project_completion():
    """Calculate overall project completion percentage"""
    
    print("\n🎯 PROJECT COMPLETION ASSESSMENT")
    print("-" * 40)
    
    # Define completion criteria
    criteria = {
        "Project structure": Path("src").exists() and Path("data").exists(),
        "Source code": Path("src/flood_susceptibility_cnn.py").exists(),
        "Raw data": Path("data/raw/Tahirpur").exists() and len(list(Path("data/raw/Tahirpur").glob("*.tif"))) >= 18,
        "Dependencies": True,  # Simplified for now
        "Data preprocessing": Path("data/processed").exists() and len(list(Path("data/processed").glob("*.npy"))) > 0,
        "Model training": Path("models/trained").exists() and len(list(Path("models/trained").glob("*.h5"))) > 0,
        "Results generation": Path("outputs/maps").exists() and len(list(Path("outputs/maps").glob("*.tif"))) > 0,
        "Documentation": Path("docs/README_CNN_Implementation.md").exists()
    }
    
    completed = sum(1 for status in criteria.values() if status)
    total = len(criteria)
    completion_pct = (completed / total) * 100
    
    for task, status in criteria.items():
        status_icon = "✅" if status else "⏳"
        print(f"{status_icon} {task}")
    
    print(f"\n📊 Overall completion: {completion_pct:.1f}% ({completed}/{total})")
    
    # Phase assessment
    if completion_pct >= 80:
        phase = "🚀 Ready for publication"
    elif completion_pct >= 60:
        phase = "🧠 Model training phase"
    elif completion_pct >= 40:
        phase = "📊 Data processing phase"
    else:
        phase = "🏗️ Setup phase"
    
    print(f"🎪 Current phase: {phase}")
    
    return completion_pct

def provide_next_steps(completion_pct):
    """Provide specific next steps based on current status"""
    
    print("\n🎯 RECOMMENDED NEXT STEPS")
    print("-" * 40)
    
    if completion_pct < 40:
        print("1. Run: python scripts/setup_project.py")
        print("2. Install dependencies: pip install -r config/requirements.txt")
        print("3. Run: python src/data_exploration.py")
    elif completion_pct < 60:
        print("1. Run: python src/flood_susceptibility_cnn.py")
        print("2. Check data preprocessing outputs in data/processed/")
        print("3. Begin CNN model implementation")
    elif completion_pct < 80:
        print("1. Complete CNN model training")
        print("2. Evaluate model performance")
        print("3. Generate flood susceptibility maps")
    else:
        print("1. Finalize results analysis")
        print("2. Prepare research documentation")
        print("3. Consider Phase 2: HEC-RAS integration")

def main():
    """Main status check function"""
    
    print("🌊 CNN FLOOD SUSCEPTIBILITY MAPPING - PROJECT STATUS")
    print("=" * 65)
    
    # Check all components
    check_project_structure()
    check_source_code()
    check_data_status()
    check_model_status()
    check_outputs()
    check_dependencies()
    get_git_status()
    
    # Calculate completion and provide guidance
    completion_pct = calculate_project_completion()
    provide_next_steps(completion_pct)
    
    print("\n" + "=" * 65)
    print("📞 For issues or questions, refer to docs/README_CNN_Implementation.md")
    print("=" * 65)

if __name__ == "__main__":
    main()