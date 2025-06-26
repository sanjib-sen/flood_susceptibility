## 🌊 Project Overview

This project uses a Deep Learning approach (Convolutional Neural Networks - CNN) to create high-resolution flood risk maps for Tahirpur, Bangladesh. The project is **COMPLETED** and the CNN model achieved 99.9% AUC performance.

## 🚀 Quick Start

### 1\. Installation

You'll need Python 3.8+ and an NVIDIA GPU with CUDA 12.x for optimal performance.

**Recommended (UV Package Manager):**

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the project
git clone <repository-url>
cd Tahirpur

# Install dependencies
uv sync
```

**Standard Python Installation:**

```bash
# Clone the project
git clone <repository-url>
cd Tahirpur

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies (with GPU support)
pip install tensorflow[and-cuda] rasterio scikit-learn matplotlib pandas
```

For CPU-only, omit `[and-cuda]` from the `tensorflow` installation.

### 2\. Run Flood Mapping

```bash
# Activate your environment (if not already active)
source .venv/bin/activate

# Run the flood mapping process
python src/cnn_flood_susceptibility.py
```

This will take about 1-2 minutes with a compatible GPU.

## 📊 Model Results

The CNN model achieved a **99.09% AUC performance**.

The model identifies the following flood factors as most important for flood susceptibility:

1.  **Elevation** (21.2% importance)
2.  **NDWI** (17.7% importance)
3.  **Slope** (16.9% importance)
4.  **Distance from Waterbody** (15.0% importance)

**Generated Output Files:**
The results are saved in the `outputs/maps/` directory:

  * `flood_susceptibility_continuous.tif`: A map showing continuous flood risk from 0 (lowest) to 1 (highest).
  * `flood_susceptibility_classified.tif`: A map classifying risk into 5 levels (Very Low, Low, Moderate, High, Very High).
  * `flood_susceptibility_visualization.png`: A combined image for easy viewing.
  * `factor_importance.csv`: A file listing the importance of each flood factor.

## 🧠 Methodology

This project uses a **Convolutional Neural Network (CNN)**, a type of deep learning model, to predict flood susceptibility. The model learns complex relationships from 8 flood-related factors (like elevation, slope, and rainfall) to identify flood-prone areas. The CNN automatically learns spatial patterns and non-linear relationships within the data, leading to higher accuracy compared to traditional methods.
