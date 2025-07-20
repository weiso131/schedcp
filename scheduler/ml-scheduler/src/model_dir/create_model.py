#!/usr/bin/env python3
"""
Create a sample model_stress_model.h5 for testing
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_stress_model():
    """Create the stress model as used in the original implementation."""
    model = keras.Sequential([
        keras.layers.Input(shape=(5,), name='input'),  # 5 features: cpu, cpu_idle, cpu_not_idle, src_dom_load, dst_dom_load
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', name='output')  # Binary output: migrate or not
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_training_data(n_samples=10000):
    """Generate synthetic training data based on migration logic."""
    np.random.seed(42)
    
    # Generate features
    cpu = np.random.randint(0, 128, n_samples)
    cpu_idle = np.random.randint(0, 64, n_samples)
    cpu_not_idle = np.random.randint(0, 64, n_samples)
    src_dom_load = np.random.uniform(0, 1, n_samples)
    dst_dom_load = np.random.uniform(0, 1, n_samples)
    
    X = np.column_stack([cpu, cpu_idle, cpu_not_idle, src_dom_load, dst_dom_load])
    
    # Migration logic: migrate if destination has significantly lower load
    # and there are idle CPUs available
    load_diff = src_dom_load - dst_dom_load
    should_migrate = (load_diff > 0.3) & (cpu_idle > 0) & (dst_dom_load < 0.7)
    
    # Add some noise to make it more realistic
    noise = np.random.random(n_samples) < 0.1
    y = (should_migrate ^ noise).astype(np.float32)
    
    return X, y

def main():
    print("Creating stress model...")
    model = create_stress_model()
    
    print("Generating training data...")
    X_train, y_train = generate_training_data()
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    print("Saving model...")
    model.save('model_stress_model.h5')
    print("Model saved as model_stress_model.h5")
    
    # Test the model
    test_input = np.array([[32, 16, 16, 0.8, 0.3]])  # High source load, low dest load
    prediction = model.predict(test_input)
    print(f"\nTest prediction: {prediction[0][0]:.3f} (should be close to 1.0 for migration)")

if __name__ == "__main__":
    main()