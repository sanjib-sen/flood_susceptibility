#!/usr/bin/env python3
"""
Full CNN Training for Flood Susceptibility - All 18 Factors
Comprehensive implementation using all harmonized factors
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class FullCNNTrainer:
    def __init__(self):
        self.harmonized_dir = Path("data/processed/harmonized_full")
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # All 18 harmonized factors
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
        
        print(f"🎯 Training CNN with all {len(self.factors)} harmonized factors")
    
    def load_all_harmonized_factors(self):
        """Load all 18 harmonized factors"""
        print("📊 Loading all harmonized factors...")
        
        factor_arrays = []
        reference_profile = None
        factor_names = []
        
        for factor in self.factors:
            filepath = self.harmonized_dir / factor
            if filepath.exists():
                try:
                    with rasterio.open(filepath) as src:
                        data = src.read(1)
                        factor_arrays.append(data)
                        factor_names.append(factor)
                        
                        if reference_profile is None:
                            reference_profile = src.profile.copy()
                        
                        # Calculate basic statistics
                        valid_data = data[~np.isnan(data)]
                        if src.nodata is not None:
                            valid_data = data[data != src.nodata]
                        
                        if len(valid_data) > 0:
                            print(f"✅ {factor:<35} Shape: {data.shape}, Valid: {len(valid_data):,}, Range: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}]")
                        else:
                            print(f"⚠️  {factor:<35} Shape: {data.shape}, No valid data")
                            
                except Exception as e:
                    print(f"❌ Failed to load {factor}: {e}")
            else:
                print(f"❌ Missing: {factor}")
        
        if len(factor_arrays) == 0:
            raise ValueError("No harmonized factors found!")
        
        # Stack all factors
        factor_stack = np.stack(factor_arrays, axis=-1)
        print(f"\n📦 Complete factor stack: {factor_stack.shape}")
        print(f"🎯 Using {len(factor_names)} factors for training")
        
        return factor_stack, reference_profile, factor_names
    
    def create_improved_flood_labels(self, factor_stack, factor_names):
        """Create improved flood susceptibility labels using multiple factors"""
        print("🏷️  Creating improved flood susceptibility labels...")
        
        # Get factor indices
        factor_dict = {name: i for i, name in enumerate(factor_names)}
        
        # Extract key factors for labeling
        try:
            elevation = factor_stack[:, :, factor_dict['Elevation.tif']]
            ndwi = factor_stack[:, :, factor_dict['NDWI_Taherpur_2025.tif']]
            slope = factor_stack[:, :, factor_dict['Slope.tif']]
            rainfall = factor_stack[:, :, factor_dict['Rainfall.tif']]
            distance_water = factor_stack[:, :, factor_dict['Distance_from_Waterbody.tif']]
            
            print("✅ Key factors extracted for labeling")
            
        except KeyError as e:
            print(f"❌ Missing key factor for labeling: {e}")
            # Fallback to available factors
            elevation = factor_stack[:, :, 0]  # First factor
            ndwi = factor_stack[:, :, -1]      # Last factor
            slope = factor_stack[:, :, 1] if factor_stack.shape[2] > 1 else elevation
            rainfall = elevation  # Fallback
            distance_water = elevation  # Fallback
            print("⚠️  Using fallback factors for labeling")
        
        # Normalize factors
        def safe_normalize(data):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                return np.zeros_like(data)
            
            data_min, data_max = np.nanmin(valid_data), np.nanmax(valid_data)
            if data_max == data_min:
                return np.zeros_like(data)
            
            normalized = (data - data_min) / (data_max - data_min)
            normalized[np.isnan(data)] = 0
            return normalized
        
        elev_norm = safe_normalize(elevation)
        ndwi_norm = safe_normalize(ndwi)
        slope_norm = safe_normalize(slope)
        rain_norm = safe_normalize(rainfall)
        dist_norm = safe_normalize(distance_water)
        
        # Improved flood susceptibility formula
        # Higher susceptibility = low elevation + high water index + low slope + high rainfall + close to water
        flood_susceptibility = (
            (1 - elev_norm) * 0.30 +       # Low elevation (30%)
            ndwi_norm * 0.25 +              # High water index (25%)
            (1 - slope_norm) * 0.20 +       # Low slope (20%)
            rain_norm * 0.15 +              # High rainfall (15%)
            (1 - dist_norm) * 0.10          # Close to water bodies (10%)
        )
        
        # Handle NaN values
        flood_susceptibility[np.isnan(flood_susceptibility)] = 0
        
        # Create labels using multiple thresholds
        very_high_thresh = np.nanpercentile(flood_susceptibility[flood_susceptibility > 0], 90)
        high_thresh = np.nanpercentile(flood_susceptibility[flood_susceptibility > 0], 75)
        
        # Binary classification: top 25% as flood-prone
        binary_labels = (flood_susceptibility >= high_thresh).astype(int)
        
        # Multi-class classification
        multi_labels = np.zeros_like(flood_susceptibility, dtype=int)
        multi_labels[flood_susceptibility >= high_thresh] = 1      # High susceptibility
        multi_labels[flood_susceptibility >= very_high_thresh] = 2 # Very high susceptibility
        
        print(f"📈 Binary class distribution:")
        unique, counts = np.unique(binary_labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"   Class {u}: {c:,} pixels ({c/binary_labels.size*100:.1f}%)")
        
        print(f"📈 Multi-class distribution:")
        unique, counts = np.unique(multi_labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"   Class {u}: {c:,} pixels ({c/multi_labels.size*100:.1f}%)")
        
        return binary_labels, multi_labels, flood_susceptibility
    
    def prepare_comprehensive_training_data(self, factor_stack, labels, sample_size=50000):
        """Prepare comprehensive training data with more samples"""
        print(f"🎲 Preparing training data with {sample_size:,} samples...")
        
        # Get valid pixels (all factors have data)
        valid_mask = ~np.isnan(factor_stack).any(axis=-1)
        valid_indices = np.where(valid_mask)
        
        total_valid = len(valid_indices[0])
        print(f"📊 Total valid pixels: {total_valid:,}")
        
        # Sample data
        if total_valid > sample_size:
            sample_idx = np.random.choice(total_valid, sample_size, replace=False)
            sample_rows = valid_indices[0][sample_idx]
            sample_cols = valid_indices[1][sample_idx]
        else:
            sample_rows = valid_indices[0]
            sample_cols = valid_indices[1]
            print(f"⚠️  Using all {total_valid:,} valid pixels (less than requested {sample_size:,})")
        
        # Extract features and labels
        X = factor_stack[sample_rows, sample_cols]
        y = labels[sample_rows, sample_cols]
        
        # Remove any remaining NaN values
        final_valid = ~np.isnan(X).any(axis=1)
        X = X[final_valid]
        y = y[final_valid]
        
        print(f"🎯 Final training data: X={X.shape}, y={y.shape}")
        
        # Feature scaling
        print("📊 Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"📊 Scaled feature statistics:")
        print(f"   Mean: {np.mean(X_scaled, axis=0)[:5]}... (first 5 factors)")
        print(f"   Std:  {np.std(X_scaled, axis=0)[:5]}... (first 5 factors)")
        
        # Save scaler for later use
        import joblib
        scaler_path = self.models_dir / "feature_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"💾 Feature scaler saved: {scaler_path}")
        
        return X_scaled, y, scaler
    
    def build_comprehensive_cnn(self, input_shape, model_type='feed_forward'):
        """Build comprehensive CNN model"""
        print(f"🧠 Building {model_type} model for input shape: {input_shape}")
        
        if model_type == 'feed_forward':
            # Enhanced feed-forward network
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
        else:  # spatial_cnn
            # True 2D CNN (requires reshaping data)
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape((1, 1, input_shape[0])),
                tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Conv2D(32, (1, 1), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        
        # Compile with improved settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        print("✅ Model compiled successfully")
        model.summary()
        
        return model
    
    def train_comprehensive_model(self, model, X, y, validation_split=0.2, epochs=50):
        """Train the comprehensive CNN model"""
        print(f"🚀 Starting comprehensive training for {epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        print(f"📊 Data splits:")
        print(f"   Training: {X_train.shape}")
        print(f"   Validation: {X_val.shape}")
        print(f"   Train class distribution: {np.bincount(y_train)}")
        print(f"   Val class distribution: {np.bincount(y_val)}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.models_dir / "best_model.h5",
                monitor='val_auc', mode='max', save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation
        print("\n📈 FINAL EVALUATION")
        print("-" * 50)
        
        val_predictions = (model.predict(X_val) > 0.5).astype(int).flatten()
        val_probs = model.predict(X_val).flatten()
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_val, val_predictions)
        precision = precision_score(y_val, val_predictions)
        recall = recall_score(y_val, val_predictions)
        f1 = f1_score(y_val, val_predictions)
        auc = roc_auc_score(y_val, val_probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC:       {auc:.4f}")
        print(f"   Epochs:    {len(history.history['loss'])}")
        
        # Compare to Wang et al. benchmark
        wang_auc = 0.937
        if auc >= wang_auc:
            print(f"🎉 EXCEEDED Wang et al. benchmark! ({auc:.4f} >= {wang_auc})")
        else:
            print(f"📊 Below Wang et al. benchmark ({auc:.4f} < {wang_auc})")
        
        return model, history, metrics
    
    def save_comprehensive_results(self, model, metrics, factor_names, history):
        """Save all training results comprehensively"""
        
        # Save final model
        model_path = self.models_dir / "full_18_factor_cnn.h5"
        model.save(model_path)
        
        # Save detailed metrics
        metrics_path = self.models_dir / "comprehensive_training_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("COMPREHENSIVE CNN FLOOD SUSCEPTIBILITY TRAINING RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Study Area: Tahirpur, Bangladesh\n")
            f.write(f"Total Factors: {len(factor_names)}\n")
            f.write(f"Model Architecture: Enhanced Feed-Forward CNN\n")
            f.write(f"Reference: Wang et al. (2019) AUC benchmark = 0.937\n\n")
            
            f.write("FACTOR LIST:\n")
            f.write("-" * 30 + "\n")
            for i, factor in enumerate(factor_names, 1):
                f.write(f"{i:2d}. {factor}\n")
            
            f.write(f"\nPERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
            
            f.write(f"\nBENCHMARK COMPARISON:\n")
            f.write("-" * 30 + "\n")
            wang_auc = 0.937
            if metrics['auc'] >= wang_auc:
                f.write(f"✅ EXCEEDED Wang et al. benchmark!\n")
                f.write(f"   Our AUC: {metrics['auc']:.4f}\n")
                f.write(f"   Wang AUC: {wang_auc:.4f}\n")
                f.write(f"   Improvement: +{(metrics['auc'] - wang_auc):.4f}\n")
            else:
                f.write(f"📊 Below Wang et al. benchmark\n")
                f.write(f"   Our AUC: {metrics['auc']:.4f}\n")
                f.write(f"   Wang AUC: {wang_auc:.4f}\n")
                f.write(f"   Gap: -{(wang_auc - metrics['auc']):.4f}\n")
        
        # Save training history plot
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # AUC plot
        plt.subplot(1, 3, 3)
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.axhline(y=0.937, color='r', linestyle='--', label='Wang et al. benchmark')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.models_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Comprehensive results saved:")
        print(f"   Model: {model_path}")
        print(f"   Metrics: {metrics_path}")
        print(f"   Training plot: {plot_path}")
        
        return model_path, metrics_path, plot_path
    
    def run_comprehensive_training(self):
        """Run complete comprehensive training pipeline"""
        print("🌊 COMPREHENSIVE CNN FLOOD SUSCEPTIBILITY TRAINING")
        print("=" * 80)
        print("🎯 Using ALL 18 harmonized factors")
        print("📊 Enhanced architecture and training")
        print("🏆 Targeting Wang et al. AUC benchmark (0.937)")
        print("=" * 80)
        
        try:
            # Load all harmonized data
            factor_stack, reference_profile, factor_names = self.load_all_harmonized_factors()
            
            # Create improved labels
            binary_labels, multi_labels, susceptibility_map = self.create_improved_flood_labels(
                factor_stack, factor_names
            )
            
            # Prepare comprehensive training data
            X, y, scaler = self.prepare_comprehensive_training_data(
                factor_stack, binary_labels, sample_size=100000
            )
            
            # Build and train model
            model = self.build_comprehensive_cnn(X.shape[1:])
            model, history, metrics = self.train_comprehensive_model(model, X, y, epochs=100)
            
            # Save results
            model_path, metrics_path, plot_path = self.save_comprehensive_results(
                model, metrics, factor_names, history
            )
            
            print("\n🎉 COMPREHENSIVE TRAINING COMPLETE!")
            print("=" * 80)
            print("✅ All 18 factors processed")
            print("✅ Enhanced CNN architecture trained")
            print("✅ Comprehensive evaluation completed")
            print("✅ Results saved with benchmarking")
            
            if metrics['auc'] >= 0.937:
                print("🏆 ACHIEVED Wang et al. benchmark!")
            
            print(f"\n📈 Final Performance:")
            print(f"   AUC Score: {metrics['auc']:.4f}")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
            
            print("=" * 80)
            
            return True, model, metrics
            
        except Exception as e:
            print(f"❌ Comprehensive training failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None

if __name__ == "__main__":
    trainer = FullCNNTrainer()
    success, model, metrics = trainer.run_comprehensive_training()
    sys.exit(0 if success else 1)