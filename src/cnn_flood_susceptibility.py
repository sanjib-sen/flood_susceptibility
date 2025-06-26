#!/usr/bin/env python3
"""
CNN-Based Flood Susceptibility Mapping for Tahirpur, Bangladesh
Deep Learning Approach for Flood Risk Assessment
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.enums import Resampling
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import time

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

class CNNFloodMapper:
    def __init__(self):
        self.raw_dir = Path("data/raw/Tahirpur")
        self.outputs_dir = Path("outputs/maps")
        self.models_dir = Path("models/trained")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Flood influencing factors
        self.flood_factors = [
            'Elevation.tif',
            'Slope.tif', 
            'Rainfall.tif',
            'NDVI_Taherpur_2025.tif',
            'NDWI_Taherpur_2025.tif',
            'NDBI_Taherpur_2025.tif',
            'Distance_from_Waterbody.tif',
            'Drainage_density.tif'
        ]
        
        print("CNN-Based Flood Susceptibility Mapping")
        print(f"Processing {len(self.flood_factors)} flood factors")
    
    def get_reference_data(self):
        """Get reference coordinate system and shape"""
        elevation_path = self.raw_dir / "Elevation.tif"
        
        with rasterio.open(elevation_path) as src:
            reference_shape = src.shape
            reference_profile = src.profile.copy()
            
        return reference_shape, reference_profile
    
    def load_flood_factors(self, target_shape):
        """Load and process flood factors"""
        print("Loading flood factors...")
        
        factor_arrays = []
        factor_names = []
        
        for factor in self.flood_factors:
            filepath = self.raw_dir / factor
            
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        data = src.read(
                            1,
                            out_shape=target_shape,
                            resampling=Resampling.bilinear
                        ).astype(np.float32)
                        
                        if src.nodata is not None:
                            data[data == src.nodata] = np.nan
                        
                        valid_data = data[~np.isnan(data)]
                        valid_pct = len(valid_data) / data.size * 100
                        
                        if len(valid_data) > 100:
                            factor_arrays.append(data)
                            factor_names.append(factor)
                            print(f"  {factor}: {valid_pct:.1f}% valid data")
                        
                except Exception as e:
                    print(f"  Error loading {factor}: {e}")
        
        factor_stack = np.stack(factor_arrays, axis=-1)
        print(f"Factor stack: {factor_stack.shape}")
        
        return factor_stack, factor_names
    
    def create_flood_labels(self, factor_stack, factor_names):
        """Create flood susceptibility labels based on geomorphological factors"""
        print("Creating flood susceptibility labels...")
        
        factor_dict = {name: i for i, name in enumerate(factor_names)}
        susceptibility = np.zeros(factor_stack.shape[:2], dtype=np.float32)
        
        # Elevation factor (30% weight)
        if 'Elevation.tif' in factor_dict:
            elevation = factor_stack[:, :, factor_dict['Elevation.tif']]
            valid_mask = ~np.isnan(elevation)
            
            if np.sum(valid_mask) > 0:
                elev_norm = np.zeros_like(elevation)
                valid_elev = elevation[valid_mask]
                elev_min, elev_max = np.percentile(valid_elev, [5, 95])
                elev_norm[valid_mask] = 1 - np.clip((elevation[valid_mask] - elev_min) / (elev_max - elev_min), 0, 1)
                susceptibility += elev_norm * 0.30
        
        # Water content factor (25% weight)
        if 'NDWI_Taherpur_2025.tif' in factor_dict:
            ndwi = factor_stack[:, :, factor_dict['NDWI_Taherpur_2025.tif']]
            valid_mask = ~np.isnan(ndwi)
            
            if np.sum(valid_mask) > 0:
                ndwi_norm = np.zeros_like(ndwi)
                valid_ndwi = ndwi[valid_mask]
                ndwi_min, ndwi_max = np.percentile(valid_ndwi, [5, 95])
                ndwi_norm[valid_mask] = np.clip((ndwi[valid_mask] - ndwi_min) / (ndwi_max - ndwi_min), 0, 1)
                susceptibility += ndwi_norm * 0.25
        
        # Slope factor (20% weight)
        if 'Slope.tif' in factor_dict:
            slope = factor_stack[:, :, factor_dict['Slope.tif']]
            valid_mask = ~np.isnan(slope)
            
            if np.sum(valid_mask) > 0:
                slope_norm = np.zeros_like(slope)
                valid_slope = slope[valid_mask]
                slope_min, slope_max = np.percentile(valid_slope, [5, 95])
                slope_norm[valid_mask] = 1 - np.clip((slope[valid_mask] - slope_min) / (slope_max - slope_min), 0, 1)
                susceptibility += slope_norm * 0.20
        
        # Distance to water (15% weight)
        if 'Distance_from_Waterbody.tif' in factor_dict:
            distance = factor_stack[:, :, factor_dict['Distance_from_Waterbody.tif']]
            valid_mask = ~np.isnan(distance)
            
            if np.sum(valid_mask) > 0:
                dist_norm = np.zeros_like(distance)
                valid_dist = distance[valid_mask]
                dist_min, dist_max = np.percentile(valid_dist, [5, 95])
                dist_norm[valid_mask] = 1 - np.clip((distance[valid_mask] - dist_min) / (dist_max - dist_min), 0, 1)
                susceptibility += dist_norm * 0.15
        
        # Vegetation factor (10% weight)
        if 'NDVI_Taherpur_2025.tif' in factor_dict:
            ndvi = factor_stack[:, :, factor_dict['NDVI_Taherpur_2025.tif']]
            valid_mask = ~np.isnan(ndvi)
            
            if np.sum(valid_mask) > 0:
                ndvi_norm = np.zeros_like(ndvi)
                valid_ndvi = ndvi[valid_mask]
                ndvi_min, ndvi_max = np.percentile(valid_ndvi, [5, 95])
                ndvi_norm[valid_mask] = 1 - np.clip((ndvi[valid_mask] - ndvi_min) / (ndvi_max - ndvi_min), 0, 1)
                susceptibility += ndvi_norm * 0.10
        
        # Create binary labels
        valid_overall = ~np.isnan(factor_stack).all(axis=-1)
        susceptibility[~valid_overall] = np.nan
        
        valid_susceptibility = susceptibility[~np.isnan(susceptibility)]
        
        if len(valid_susceptibility) > 0:
            threshold = np.percentile(valid_susceptibility, 75)
            binary_labels = (susceptibility >= threshold).astype(np.int32)
            binary_labels[np.isnan(susceptibility)] = -1
            
            flood_count = np.sum(binary_labels == 1)
            non_flood_count = np.sum(binary_labels == 0)
            print(f"Labels created: {flood_count:,} flood-prone, {non_flood_count:,} non-flood")
            
            return susceptibility, binary_labels
        else:
            raise ValueError("No valid susceptibility data generated")
    
    def prepare_training_data(self, factor_stack, labels, sample_size=50000):
        """Prepare training data for CNN"""
        print(f"Preparing training data ({sample_size:,} samples)...")
        
        valid_mask = ~np.isnan(factor_stack).any(axis=-1) & (labels >= 0)
        valid_indices = np.where(valid_mask)
        total_valid = len(valid_indices[0])
        
        print(f"Total valid pixels: {total_valid:,}")
        
        if total_valid < 1000:
            raise ValueError(f"Insufficient training data: {total_valid} valid pixels")
        
        # Balanced sampling
        flood_mask = valid_mask & (labels == 1)
        non_flood_mask = valid_mask & (labels == 0)
        
        flood_indices = np.where(flood_mask)
        non_flood_indices = np.where(non_flood_mask)
        
        n_flood = len(flood_indices[0])
        n_non_flood = len(non_flood_indices[0])
        
        samples_per_class = min(sample_size // 2, n_flood, n_non_flood)
        
        flood_sample_idx = np.random.choice(n_flood, samples_per_class, replace=False)
        flood_rows = flood_indices[0][flood_sample_idx]
        flood_cols = flood_indices[1][flood_sample_idx]
        
        non_flood_sample_idx = np.random.choice(n_non_flood, samples_per_class, replace=False)
        non_flood_rows = non_flood_indices[0][non_flood_sample_idx]
        non_flood_cols = non_flood_indices[1][non_flood_sample_idx]
        
        sample_rows = np.concatenate([flood_rows, non_flood_rows])
        sample_cols = np.concatenate([flood_cols, non_flood_cols])
        
        X = factor_stack[sample_rows, sample_cols].astype(np.float32)
        y = labels[sample_rows, sample_cols].astype(np.float32)
        
        print(f"Training data: X={X.shape}, y={y.shape}")
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    
    def build_cnn_model(self, input_shape):
        """Build CNN model architecture"""
        print(f"Building CNN model for input shape: {input_shape}")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        print("CNN model compiled")
        return model
    
    def train_model(self, model, X, y, epochs=50):
        """Train the CNN model"""
        print(f"Training CNN model for {epochs} epochs...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=10, restore_best_weights=True, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model
        val_predictions = (model.predict(X_val) > 0.5).astype(int).flatten()
        val_probs = model.predict(X_val).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_predictions),
            'precision': precision_score(y_val, val_predictions),
            'recall': recall_score(y_val, val_predictions),
            'f1_score': f1_score(y_val, val_predictions),
            'auc': roc_auc_score(y_val, val_probs),
            'training_time': training_time
        }
        
        print(f"Training Results:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return model, history, metrics
    
    def extract_factor_importance(self, model, factor_names):
        """Extract factor importance from CNN model"""
        print("Extracting factor importance...")
        
        first_layer_weights = model.layers[0].get_weights()[0]
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
        priority_weights = feature_importance / np.sum(feature_importance)
        
        weights_df = pd.DataFrame({
            'Factor': factor_names,
            'Importance': priority_weights,
            'Rank': range(1, len(factor_names) + 1)
        })
        
        weights_df = weights_df.sort_values('Importance', ascending=False)
        weights_df['Rank'] = range(1, len(factor_names) + 1)
        
        print("Factor Importance Ranking:")
        for _, row in weights_df.iterrows():
            print(f"  {row['Rank']:2d}. {row['Factor']:<30} - {row['Importance']:.3f}")
        
        weights_path = self.outputs_dir / "factor_importance.csv"
        weights_df.to_csv(weights_path, index=False)
        print(f"Factor importance saved: {weights_path}")
        
        return weights_df
    
    def generate_susceptibility_map(self, model, scaler, factor_stack, reference_profile, factor_names):
        """Generate flood susceptibility map"""
        print("Generating flood susceptibility map...")
        
        rows, cols, n_factors = factor_stack.shape
        
        valid_mask = ~np.isnan(factor_stack).any(axis=-1)
        valid_indices = np.where(valid_mask)
        
        print(f"Predicting for {len(valid_indices[0]):,} valid pixels...")
        
        valid_data = factor_stack[valid_indices]
        
        try:
            if hasattr(scaler, 'mean_'):
                valid_data_scaled = scaler.transform(valid_data)
            else:
                print("Fitting scaler to current data...")
                valid_data_scaled = scaler.fit_transform(valid_data)
        except:
            print("Using manual feature scaling...")
            valid_data_scaled = (valid_data - np.mean(valid_data, axis=0)) / (np.std(valid_data, axis=0) + 1e-8)
        
        predictions = model.predict(valid_data_scaled, batch_size=1000, verbose=0)
        predictions = predictions.flatten()
        
        susceptibility_map = np.full((rows, cols), np.nan, dtype=np.float32)
        susceptibility_map[valid_indices] = predictions
        
        print(f"Susceptibility map generated: range [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        
        return susceptibility_map
    
    def create_classification(self, susceptibility_map):
        """Create 5-level risk classification"""
        print("Creating risk classification...")
        
        valid_data = susceptibility_map[~np.isnan(susceptibility_map)]
        
        if len(valid_data) == 0:
            raise ValueError("No valid data for classification")
        
        thresholds = [
            np.percentile(valid_data, 20),
            np.percentile(valid_data, 40),
            np.percentile(valid_data, 60),
            np.percentile(valid_data, 80),
        ]
        
        classified = np.full_like(susceptibility_map, 0, dtype=np.uint8)
        classified[np.isnan(susceptibility_map)] = 255
        
        for i, threshold in enumerate(thresholds):
            classified[susceptibility_map >= threshold] = i + 1
        
        print("Risk Classification Distribution:")
        class_names = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        total_valid = np.sum(classified != 255)
        
        for class_id in range(1, 6):
            count = np.sum(classified == class_id)
            percent = count / total_valid * 100 if total_valid > 0 else 0
            print(f"  {class_names[class_id-1]}: {count:,} pixels ({percent:.1f}%)")
        
        return classified
    
    def save_results(self, susceptibility_map, classified_map, reference_profile):
        """Save flood susceptibility results"""
        print("Saving results...")
        
        # Continuous susceptibility map
        continuous_path = self.outputs_dir / "flood_susceptibility_continuous.tif"
        continuous_profile = reference_profile.copy()
        continuous_profile.update(dtype='float32', nodata=np.nan)
        
        with rasterio.open(continuous_path, 'w', **continuous_profile) as dst:
            dst.write(susceptibility_map, 1)
        
        # Classified risk map
        classified_path = self.outputs_dir / "flood_susceptibility_classified.tif"
        classified_profile = reference_profile.copy()
        classified_profile.update(dtype='uint8', nodata=255)
        
        with rasterio.open(classified_path, 'w', **classified_profile) as dst:
            dst.write(classified_map, 1)
        
        print(f"Continuous map: {continuous_path}")
        print(f"Classified map: {classified_path}")
        
        return continuous_path, classified_path
    
    def create_visualization(self, susceptibility_map, classified_map, reference_profile):
        """Create flood susceptibility visualization"""
        print("Creating visualization...")
        
        # Get coordinate bounds for display
        transform = reference_profile['transform']
        height, width = susceptibility_map.shape
        
        left = transform[2]
        top = transform[5]
        right = left + width * transform[0]
        bottom = top + height * transform[4]
        
        # Approximate lat/lon conversion for Bangladesh UTM Zone 45N
        approx_lon_left = (left - 500000) / 111320 + 93
        approx_lon_right = (right - 500000) / 111320 + 93
        approx_lat_bottom = bottom / 111320
        approx_lat_top = top / 111320
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Continuous susceptibility
        ax1 = axes[0]
        susceptibility_masked = np.ma.masked_invalid(susceptibility_map)
        
        im1 = ax1.imshow(susceptibility_masked, cmap='viridis', vmin=0, vmax=1, 
                        extent=[approx_lon_left, approx_lon_right, approx_lat_bottom, approx_lat_top],
                        interpolation='bilinear')
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
        cbar1.set_label('Susceptibility Index', fontsize=10)
        cbar1.ax.tick_params(labelsize=8)
        
        ax1.set_title('Flood Susceptibility\\n(Continuous)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Longitude', fontsize=10)
        ax1.set_ylabel('Latitude', fontsize=10)
        ax1.tick_params(labelsize=8)
        
        # Risk classification
        ax2 = axes[1]
        colors = ['#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
        class_cmap = ListedColormap(colors)
        
        classified_display = classified_map.copy().astype(float)
        classified_display[classified_map == 255] = np.nan
        classified_masked = np.ma.masked_invalid(classified_display)
        
        im2 = ax2.imshow(classified_masked, cmap=class_cmap, vmin=1, vmax=5,
                        extent=[approx_lon_left, approx_lon_right, approx_lat_bottom, approx_lat_top],
                        interpolation='nearest')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, ticks=[1, 2, 3, 4, 5])
        cbar2.set_ticklabels(['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        cbar2.set_label('Risk Level', fontsize=10)
        cbar2.ax.tick_params(labelsize=8)
        
        ax2.set_title('Flood Risk Classification\\n(5 Levels)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Longitude', fontsize=10)
        ax2.set_ylabel('Latitude', fontsize=10)
        ax2.tick_params(labelsize=8)
        
        # Main title
        fig.suptitle('CNN-Based Flood Susceptibility Mapping\\nTahirpur, Bangladesh', 
                     fontsize=12, fontweight='bold')
        
        # Coordinate info
        fig.text(0.5, 0.02, f'UTM Zone 45N - Grid: {height}×{width}',
                ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        # Save visualization
        viz_path = self.outputs_dir / "flood_susceptibility_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization: {viz_path}")
        
        return viz_path
    
    def run_flood_mapping(self):
        """Run complete CNN flood susceptibility mapping"""
        print("CNN-Based Flood Susceptibility Mapping")
        print("=" * 50)
        print("Deep Learning Approach for Flood Risk Assessment")
        print("=" * 50)
        
        try:
            # Get reference data
            reference_shape, reference_profile = self.get_reference_data()
            
            # Load flood factors
            factor_stack, factor_names = self.load_flood_factors(reference_shape)
            
            # Create labels
            susceptibility, labels = self.create_flood_labels(factor_stack, factor_names)
            
            # Prepare training data
            X, y, scaler = self.prepare_training_data(factor_stack, labels)
            
            # Build and train model
            model = self.build_cnn_model(X.shape[1:])
            model, history, metrics = self.train_model(model, X, y)
            
            # Extract factor importance
            weights_df = self.extract_factor_importance(model, factor_names)
            
            # Generate susceptibility map
            spatial_map = self.generate_susceptibility_map(model, scaler, factor_stack, reference_profile, factor_names)
            
            # Create classification
            classified_map = self.create_classification(spatial_map)
            
            # Save results
            continuous_path, classified_path = self.save_results(spatial_map, classified_map, reference_profile)
            
            # Create visualization
            viz_path = self.create_visualization(spatial_map, classified_map, reference_profile)
            
            # Save model
            model_path = self.models_dir / "flood_susceptibility_model.h5"
            model.save(model_path)
            
            print("\nFlood Susceptibility Mapping Complete")
            print("=" * 50)
            print(f"Model Performance: AUC = {metrics['auc']:.3f}")
            print(f"Factor Importance: {len(weights_df)} factors ranked")
            print(f"Output Maps: Continuous + Classified")
            print("=" * 50)
            
            return True, {
                'model': model,
                'metrics': metrics,
                'importance': weights_df,
                'continuous_map': continuous_path,
                'classified_map': classified_path,
                'visualization': viz_path
            }
            
        except Exception as e:
            print(f"Flood mapping failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None

if __name__ == "__main__":
    mapper = CNNFloodMapper()
    success, results = mapper.run_flood_mapping()
    sys.exit(0 if success else 1)