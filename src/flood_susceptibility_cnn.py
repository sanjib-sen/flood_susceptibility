"""
CNN-based Flood Susceptibility Mapping for Tahirpur, Bangladesh
Based on Wang et al. (2019) methodology with 2D-CNN architecture

Author: Research Implementation
Date: 2025
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")


class FloodSusceptibilityCNN:
    """
    CNN-based flood susceptibility mapping
    """

    def __init__(
        self, data_dir="data/raw/Tahirpur/", target_resolution=30, target_crs="EPSG:32645"
    ):
        """
        Initialize the CNN flood susceptibility mapper

        Args:
            data_dir: Directory containing GIS data
            target_resolution: Target spatial resolution in meters
            target_crs: Target coordinate reference system
        """
        self.data_dir = Path(data_dir)
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.model = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

        # Define flood influencing factors based on available data
        self.continuous_factors = [
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
            "WRI_Taherpur_2025.tif",
        ]

        self.categorical_factors = [
            "Lithology.tif",
            "Export_LULC_Taherpur_Sentinel2.tif",
        ]

    def load_and_preprocess_raster(self, filepath):
        """
        Load and preprocess a single raster file

        Args:
            filepath: Path to raster file

        Returns:
            Preprocessed raster array and metadata
        """
        try:
            with rasterio.open(filepath) as src:
                # Read data
                data = src.read(1)

                # Handle nodata values
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)

                # Get metadata
                metadata = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                }

                return data, metadata

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None

    def resample_to_target_grid(
        self,
        data,
        src_transform,
        src_crs,
        target_bounds,
        target_transform,
        resampling_method=Resampling.bilinear,
    ):
        """
        Resample raster data to target grid

        Args:
            data: Source raster data
            src_transform: Source transform
            src_crs: Source CRS
            target_bounds: Target bounds
            target_transform: Target transform
            resampling_method: Resampling method

        Returns:
            Resampled data array
        """
        # Calculate target dimensions
        target_width = int(
            (target_bounds[2] - target_bounds[0]) / self.target_resolution
        )
        target_height = int(
            (target_bounds[3] - target_bounds[1]) / self.target_resolution
        )

        # Create destination array
        destination = np.empty((target_height, target_width), dtype=data.dtype)

        # Reproject
        reproject(
            source=data,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=self.target_crs,
            resampling=resampling_method,
        )

        return destination

    def create_target_grid(self, all_bounds):
        """
        Create common target grid for all rasters

        Args:
            all_bounds: List of bounds from all rasters

        Returns:
            Target bounds and transform
        """
        # Find common extent
        min_x = min([bounds[0] for bounds in all_bounds])
        min_y = min([bounds[1] for bounds in all_bounds])
        max_x = max([bounds[2] for bounds in all_bounds])
        max_y = max([bounds[3] for bounds in all_bounds])

        target_bounds = (min_x, min_y, max_x, max_y)

        # Create target transform
        target_transform = from_bounds(
            min_x,
            min_y,
            max_x,
            max_y,
            int((max_x - min_x) / self.target_resolution),
            int((max_y - min_y) / self.target_resolution),
        )

        return target_bounds, target_transform

    def load_all_factors(self):
        """
        Load and harmonize all flood influencing factors

        Returns:
            Stacked array of all factors and valid mask
        """
        print("Loading and preprocessing flood influencing factors...")

        all_factors = []
        all_bounds = []
        factor_names = []

        # Load continuous factors
        for factor in self.continuous_factors:
            filepath = self.data_dir / factor
            if filepath.exists():
                data, metadata = self.load_and_preprocess_raster(filepath)
                if data is not None:
                    all_factors.append((data, metadata, "continuous", factor))
                    all_bounds.append(metadata["bounds"])
                    factor_names.append(factor.replace(".tif", ""))
                    print(f"✓ Loaded {factor}")
                else:
                    print(f"✗ Failed to load {factor}")
            else:
                print(f"✗ File not found: {factor}")

        # Load categorical factors
        for factor in self.categorical_factors:
            filepath = self.data_dir / factor
            if filepath.exists():
                data, metadata = self.load_and_preprocess_raster(filepath)
                if data is not None:
                    all_factors.append((data, metadata, "categorical", factor))
                    all_bounds.append(metadata["bounds"])
                    factor_names.append(factor.replace(".tif", ""))
                    print(f"✓ Loaded {factor}")
                else:
                    print(f"✗ Failed to load {factor}")
            else:
                print(f"✗ File not found: {factor}")

        if not all_factors:
            raise ValueError("No factors could be loaded!")

        print(f"\nTotal factors loaded: {len(all_factors)}")

        # Create common grid
        target_bounds, target_transform = self.create_target_grid(all_bounds)
        print(f"Target grid bounds: {target_bounds}")

        # Resample all factors to common grid
        print("\nResampling factors to common grid...")
        harmonized_factors = []

        for data, metadata, factor_type, filename in all_factors:
            if factor_type == "continuous":
                resampling_method = Resampling.bilinear
            else:
                resampling_method = Resampling.nearest

            resampled = self.resample_to_target_grid(
                data,
                metadata["transform"],
                metadata["crs"],
                target_bounds,
                target_transform,
                resampling_method,
            )

            harmonized_factors.append(resampled)
            print(f"✓ Resampled {filename}")

        # Stack all factors
        factor_stack = np.stack(harmonized_factors, axis=-1)
        print(f"Factor stack shape: {factor_stack.shape}")

        # Create valid data mask (pixels with no NaN values across all bands)
        valid_mask = ~np.isnan(factor_stack).any(axis=-1)
        valid_pixels = np.sum(valid_mask)
        total_pixels = factor_stack.shape[0] * factor_stack.shape[1]

        print(
            f"Valid pixels: {valid_pixels:,} ({valid_pixels / total_pixels * 100:.1f}%)"
        )

        return factor_stack, valid_mask, factor_names, target_bounds, target_transform

    def create_training_samples(self, factor_stack, valid_mask, sample_size=10000):
        """
        Create training samples for flood susceptibility classification
        Since we don't have historical flood data, we'll use topographic and hydrological
        proxies to create pseudo-labels

        Args:
            factor_stack: Stacked factor arrays
            valid_mask: Mask of valid pixels
            sample_size: Number of samples to create

        Returns:
            Training features and labels
        """
        print("Creating training samples...")

        # Get valid pixel coordinates
        valid_coords = np.where(valid_mask)
        n_valid = len(valid_coords[0])

        # Sample random valid pixels
        if n_valid > sample_size:
            sample_indices = np.random.choice(n_valid, sample_size, replace=False)
            sample_coords = (
                valid_coords[0][sample_indices],
                valid_coords[1][sample_indices],
            )
        else:
            sample_coords = valid_coords
            print(f"Warning: Only {n_valid} valid pixels available, using all")

        # Extract features for sampled pixels
        features = factor_stack[sample_coords]

        # Create pseudo-labels based on flood susceptibility proxies
        # This is a simplified approach - in real research, you'd use historical flood data
        labels = self.create_flood_susceptibility_labels(features)

        print(f"Created {len(features)} training samples")
        print(f"Label distribution: {np.bincount(labels)}")

        return features, labels

    def create_flood_susceptibility_labels(self, features):
        """
        Create flood susceptibility labels based on multiple criteria
        This is a proxy method - replace with actual flood inventory data when available

        Args:
            features: Feature array

        Returns:
            Labels (0: Low, 1: Medium, 2: High susceptibility)
        """
        # Assuming feature order matches loaded factors
        # You'll need to adjust indices based on your actual feature order

        # Extract key flood-related features (adjust indices as needed)
        elevation = features[:, 0]  # Elevation
        slope = features[:, 1]  # Slope
        twi = features[:, 3]  # TWI
        distance_water = features[:, 6]  # Distance from water
        ndwi = features[:, 9]  # NDWI

        # Normalize features for scoring
        elevation_norm = (elevation - np.nanmin(elevation)) / (
            np.nanmax(elevation) - np.nanmin(elevation)
        )
        slope_norm = (slope - np.nanmin(slope)) / (np.nanmax(slope) - np.nanmin(slope))
        twi_norm = (twi - np.nanmin(twi)) / (np.nanmax(twi) - np.nanmin(twi))
        distance_norm = (distance_water - np.nanmin(distance_water)) / (
            np.nanmax(distance_water) - np.nanmin(distance_water)
        )
        ndwi_norm = (ndwi - np.nanmin(ndwi)) / (np.nanmax(ndwi) - np.nanmin(ndwi))

        # Create composite susceptibility score
        # Higher susceptibility for: low elevation, low slope, high TWI, near water, high NDWI
        susceptibility_score = (
            (1 - elevation_norm) * 0.3  # Lower elevation = higher flood risk
            + (1 - slope_norm) * 0.2  # Lower slope = higher flood risk
            + twi_norm * 0.25  # Higher TWI = higher flood risk
            + (1 - distance_norm) * 0.15  # Closer to water = higher flood risk
            + ndwi_norm * 0.1  # Higher water content = higher flood risk
        )

        # Convert to categorical labels using quantiles
        low_threshold = np.nanpercentile(susceptibility_score, 33.33)
        high_threshold = np.nanpercentile(susceptibility_score, 66.67)

        labels = np.zeros(len(susceptibility_score), dtype=int)
        labels[susceptibility_score > low_threshold] = 1  # Medium
        labels[susceptibility_score > high_threshold] = 2  # High

        return labels

    def normalize_features(self, features, fit_scaler=True):
        """
        Normalize features using MinMaxScaler

        Args:
            features: Feature array
            fit_scaler: Whether to fit the scaler

        Returns:
            Normalized features
        """
        if fit_scaler:
            # Handle NaN values by fitting only on valid data
            valid_mask = ~np.isnan(features).any(axis=1)
            valid_features = features[valid_mask]
            self.scaler.fit(valid_features)

        # Transform all features (NaN will remain NaN)
        normalized = np.empty_like(features)
        for i in range(features.shape[0]):
            if not np.isnan(features[i]).any():
                normalized[i] = self.scaler.transform(
                    features[i].reshape(1, -1)
                ).flatten()
            else:
                normalized[i] = features[i]  # Keep NaN as is

        return normalized

    def build_2d_cnn_model(self, input_shape, num_classes=3):
        """
        Build 2D CNN model

        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of flood susceptibility classes

        Returns:
            Compiled CNN model
        """
        model = keras.Sequential(
            [
                # First convolutional block
                layers.Conv2D(20, (3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                # Second convolutional block
                layers.Conv2D(10, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dense(50, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adagrad(learning_rate=0.0015),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def prepare_cnn_input_data(self, factor_stack, labels, patch_size=13):
        """
        Prepare input data for 2D CNN

        Args:
            factor_stack: Stacked factor arrays
            labels: Corresponding labels
            patch_size: Size of spatial patches

        Returns:
            CNN input arrays and labels
        """
        print(f"Preparing CNN input with patch size {patch_size}x{patch_size}...")

        # Convert to 2D format as described in Wang et al. paper
        # Each pixel vector is converted to a patch matrix
        n_factors = factor_stack.shape[-1]
        matrix_size = max(n_factors, patch_size)

        cnn_inputs = []
        valid_labels = []

        # Get valid pixels with enough spatial context
        height, width = factor_stack.shape[:2]
        padding = patch_size // 2

        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                # Extract spatial patch
                patch = factor_stack[
                    i - padding : i + padding + 1, j - padding : j + padding + 1, :
                ]

                # Check if patch is completely valid (no NaN)
                if not np.isnan(patch).any():
                    # Reshape to matrix format
                    # This creates a matrix where each row represents a factor
                    patch_matrix = np.zeros((matrix_size, matrix_size, 1))

                    # Fill the matrix with factor values
                    for k in range(n_factors):
                        if k < matrix_size:
                            patch_matrix[k, :patch_size, 0] = patch[:, :, k].flatten()[
                                :patch_size
                            ]

                    cnn_inputs.append(patch_matrix)
                    # Note: You'll need to create corresponding labels for spatial patches

        if len(cnn_inputs) == 0:
            print("Warning: No valid patches found, using simplified approach")
            # Fallback to pixel-based approach
            return self.prepare_pixel_based_input(factor_stack, labels)

        cnn_inputs = np.array(cnn_inputs)
        print(f"Created {len(cnn_inputs)} CNN input patches")
        print(f"Input shape: {cnn_inputs.shape}")

        return cnn_inputs, np.array(valid_labels)

    def prepare_pixel_based_input(self, factor_stack, labels):
        """
        Simplified pixel-based input preparation

        Args:
            factor_stack: Stacked factor arrays
            labels: Corresponding labels

        Returns:
            Reshaped inputs for CNN
        """
        # For simplified approach, treat each factor as a separate "channel"
        # Reshape to (samples, height, width, channels) format

        n_factors = factor_stack.shape[-1]
        # Create small patches around each pixel
        patch_size = 3  # Use smaller patches for memory efficiency

        inputs = []
        valid_labels = []

        height, width = factor_stack.shape[:2]
        padding = patch_size // 2

        # Sample a subset of pixels for training
        sample_indices = np.random.choice(
            (height - 2 * padding) * (width - 2 * padding),
            size=min(5000, (height - 2 * padding) * (width - 2 * padding)),
            replace=False,
        )

        count = 0
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                if count in sample_indices:
                    patch = factor_stack[
                        i - padding : i + padding + 1, j - padding : j + padding + 1, :
                    ]

                    if not np.isnan(patch).any():
                        inputs.append(patch)
                        # Create label for this pixel (simplified)
                        pixel_features = factor_stack[i, j, :]
                        if not np.isnan(pixel_features).any():
                            pixel_label = self.create_flood_susceptibility_labels(
                                pixel_features.reshape(1, -1)
                            )[0]
                            valid_labels.append(pixel_label)
                        else:
                            inputs.pop()  # Remove the input we just added

                count += 1

        if len(inputs) == 0:
            raise ValueError("No valid input patches created")

        inputs = np.array(inputs)
        valid_labels = np.array(valid_labels)

        print(f"Created {len(inputs)} pixel-based input patches")
        print(f"Input shape: {inputs.shape}")
        print(f"Label distribution: {np.bincount(valid_labels)}")

        return inputs, valid_labels


def main():
    """
    Main execution function
    """
    print("=== CNN-based Flood Susceptibility Mapping ===")

    # Initialize the CNN mapper
    mapper = FloodSusceptibilityCNN()

    # Load and preprocess all factors
    try:
        factor_stack, valid_mask, factor_names, target_bounds, target_transform = (
            mapper.load_all_factors()
        )

        # Save preprocessing results
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/factor_stack.npy", factor_stack)
        np.save("data/processed/valid_mask.npy", valid_mask)

        print("\n" + "=" * 50)
        print("Data preprocessing completed successfully!")
        print(f"Factor names: {factor_names}")
        print(f"Data shape: {factor_stack.shape}")
        print(f"Valid pixels: {np.sum(valid_mask):,}")
        print("=" * 50)

    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return False

    return True


if __name__ == "__main__":
    main()

