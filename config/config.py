"""
Configuration file for CNN Flood Susceptibility Mapping
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw" / "Tahirpur"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = MODELS_DIR / "logs"

# Output paths
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Model configuration
MODEL_CONFIG = {
    "target_resolution": 30,  # meters
    "target_crs": "EPSG:32645",  # UTM Zone 45N
    "patch_size": 13,  # spatial patch size for CNN
    "num_classes": 3,  # Low, Medium, High susceptibility
    "batch_size": 32,
    "epochs": 400,
    "learning_rate": 0.0015,
    "validation_split": 0.3
}

# Data configuration
DATA_CONFIG = {
    "continuous_factors": [
        "Elevation.tif",
        "Slope.tif", 
        "Curvature.tif",
        "TWI.tif",
        "TPI.tif",
        "Rainfall.tif",
        "Distance_from_Waterbody.tif",
        "Drainage_density.tif",
        "NDVI_Taherpur_2025.tif",
        "NDWI_Taherpur_2025.tif",
        "NDBI_Taherpur_2025.tif",
        "NDMI_Taherpur_2025.tif",
        "BSI_Taherpur_2025.tif",
        "MSI_Taherpur_2025.tif",
        "WRI_Taherpur_2025.tif"
    ],
    "categorical_factors": [
        "Lithology.tif",
        "Export_LULC_Taherpur_Sentinel2.tif"
    ]
}

# Output filenames
OUTPUT_FILES = {
    "factor_stack": "factor_stack.npy",
    "valid_mask": "valid_mask.npy", 
    "trained_model": "flood_susceptibility_cnn.h5",
    "model_weights": "model_weights.h5",
    "scaler": "feature_scaler.pkl",
    "classification_report": "classification_report.txt",
    "susceptibility_map": "flood_susceptibility_map.tif",
    "probability_map": "flood_probability_map.tif"
}