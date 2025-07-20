#!/usr/bin/env python3
"""
Script to export a trained Keras model to TensorFlow SavedModel format
for use with the ML scheduler.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

def create_sample_model():
    """Create a sample neural network model for task migration decisions."""
    model = keras.Sequential([
        keras.layers.Input(shape=(5,)),  # 5 input features
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_sample_data(n_samples=1000):
    """Generate synthetic training data for demonstration."""
    # Features: cpu, cpu_idle, cpu_not_idle, src_dom_load, dst_dom_load
    X = np.random.rand(n_samples, 5)
    
    # Simple rule: migrate if destination domain has significantly lower load
    y = (X[:, 4] < X[:, 3] - 0.3).astype(np.float32)
    
    return X, y

def train_and_export_model(model_path='model'):
    """Train the model and export it to SavedModel format."""
    # Create and train model
    model = create_sample_model()
    X_train, y_train = generate_sample_data()
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Export to SavedModel format
    print(f"Exporting model to {model_path}...")
    tf.saved_model.save(model, model_path)
    
    # Also save as .pb for easier loading
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    
    with open(f"{model_path}.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("Model exported successfully!")

if __name__ == "__main__":
    train_and_export_model()