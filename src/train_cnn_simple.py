#!/usr/bin/env python3
"""
Simple CNN Training for Flood Susceptibility
Focused approach using harmonized data
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class SimpleCNNTrainer:
    def __init__(self):
        self.harmonized_dir = Path("data/processed/harmonized")
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Priority factors that were harmonized
        self.factors = [
            'Elevation.tif',
            'Slope.tif',
            'Rainfall.tif',
            'NDVI_Taherpur_2025.tif',
            'NDWI_Taherpur_2025.tif'
        ]
        
        print(f"🎯 Using {len(self.factors)} harmonized factors")
    
    def load_harmonized_data(self):
        """Load harmonized raster data"""
        print("📊 Loading harmonized data...")
        
        factor_arrays = []
        
        for factor in self.factors:
            filepath = self.harmonized_dir / factor
            if filepath.exists():
                with rasterio.open(filepath) as src:
                    data = src.read(1)
                    factor_arrays.append(data)
                    print(f"✅ Loaded {factor}: {data.shape}")
            else:
                print(f"❌ Missing: {factor}")
        
        if len(factor_arrays) == 0:
            raise ValueError("No harmonized data found!")
        
        # Stack factors
        factor_stack = np.stack(factor_arrays, axis=-1)
        print(f"📦 Factor stack shape: {factor_stack.shape}")
        
        return factor_stack
    
    def create_synthetic_labels(self, factor_stack):
        """Create synthetic flood labels based on elevation and water indices"""
        print("🏷️  Creating synthetic flood labels...")
        
        # Extract relevant factors
        elevation = factor_stack[:, :, 0]  # Elevation
        ndwi = factor_stack[:, :, 4]       # NDWI (water index)
        
        # Normalize
        elev_norm = (elevation - np.nanmin(elevation)) / (np.nanmax(elevation) - np.nanmin(elevation))
        ndwi_norm = (ndwi - np.nanmin(ndwi)) / (np.nanmax(ndwi) - np.nanmin(ndwi))
        
        # Create flood susceptibility labels
        # High susceptibility: low elevation + high water index
        flood_susceptibility = (1 - elev_norm) * 0.7 + ndwi_norm * 0.3
        
        # Convert to binary classification
        threshold = np.nanpercentile(flood_susceptibility, 75)  # Top 25% as flood prone
        labels = (flood_susceptibility > threshold).astype(int)
        
        print(f"📈 Flood prone pixels: {np.sum(labels)}/{labels.size} ({np.sum(labels)/labels.size*100:.1f}%)")
        
        return labels
    
    def prepare_training_data(self, factor_stack, labels, sample_size=5000):
        """Prepare training data by sampling pixels"""
        print(f"🎲 Sampling {sample_size} pixels for training...")
        
        # Get valid pixels (no NaN values)
        valid_mask = ~np.isnan(factor_stack).any(axis=-1)
        valid_indices = np.where(valid_mask)
        
        print(f"📊 Valid pixels: {len(valid_indices[0]):,}")
        
        # Sample randomly
        if len(valid_indices[0]) > sample_size:
            sample_idx = np.random.choice(len(valid_indices[0]), sample_size, replace=False)
            sample_rows = valid_indices[0][sample_idx]
            sample_cols = valid_indices[1][sample_idx]
        else:
            sample_rows = valid_indices[0]
            sample_cols = valid_indices[1]
        
        # Extract features and labels
        X = factor_stack[sample_rows, sample_cols]
        y = labels[sample_rows, sample_cols]
        
        print(f"🎯 Training data shape: X={X.shape}, y={y.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def build_simple_cnn(self, input_shape):
        """Build a simple CNN model"""
        print(f"🧠 Building CNN model for input shape: {input_shape}")
        
        model = tf.keras.Sequential([
            # Reshape for CNN (treat features as spatial)
            tf.keras.layers.Reshape((1, 1, input_shape[0])),
            
            # Simple dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model compiled successfully")
        model.summary()
        
        return model
    
    def train_model(self, model, X, y, validation_split=0.2, epochs=20):
        """Train the CNN model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape}")
        print(f"📊 Validation set: {X_val.shape}")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        val_predictions = (model.predict(X_val) > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, val_predictions)
        precision = precision_score(y_val, val_predictions)
        recall = recall_score(y_val, val_predictions)
        f1 = f1_score(y_val, val_predictions)
        
        print(f"\n📈 FINAL METRICS:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return model, history, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, model, metrics):
        """Save the trained model"""
        model_path = self.models_dir / "simple_flood_cnn.h5"
        model.save(model_path)
        
        # Save metrics
        metrics_path = self.models_dir / "training_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Simple CNN Flood Susceptibility Training Results\n")
            f.write("=" * 50 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        print(f"💾 Model saved: {model_path}")
        print(f"📋 Metrics saved: {metrics_path}")
    
    def run_training(self):
        """Run complete training pipeline"""
        print("🌊 SIMPLE CNN FLOOD SUSCEPTIBILITY TRAINING")
        print("=" * 60)
        
        try:
            # Load data
            factor_stack = self.load_harmonized_data()
            
            # Create labels
            labels = self.create_synthetic_labels(factor_stack)
            
            # Prepare training data
            X, y = self.prepare_training_data(factor_stack, labels)
            
            # Build model
            model = self.build_simple_cnn(X.shape[1:])
            
            # Train model
            model, history, metrics = self.train_model(model, X, y)
            
            # Save results
            self.save_model(model, metrics)
            
            print("\n🎉 TRAINING COMPLETE!")
            print("=" * 60)
            print("✅ Model trained and saved")
            print("✅ Metrics recorded")
            print("🎯 Ready for flood susceptibility mapping")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

if __name__ == "__main__":
    trainer = SimpleCNNTrainer()
    success = trainer.run_training()
    sys.exit(0 if success else 1)