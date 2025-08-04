#!/usr/bin/env python3
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# Remove old model directory
if os.path.exists('model_path'):
    shutil.rmtree('model_path')

# Load the existing H5 model (which expects 5 features)
original_model = tf.keras.models.load_model('model_stress_model.h5')

# Create a new model that accepts 10 features but uses only first 5
input_layer = tf.keras.layers.Input(shape=(10,), name='input')
# Take only first 5 features
sliced = tf.keras.layers.Lambda(lambda x: x[:, :5])(input_layer)
# Pass through the original model
output = original_model(sliced)
# Create the wrapper model
wrapper_model = tf.keras.Model(inputs=input_layer, outputs=output)

# Use TensorFlow's export method
wrapper_model.export('model_path')

print("Model saved successfully to model_path/")

# Verify the saved model
print("\nVerifying saved model...")
loaded = tf.saved_model.load('model_path')

# Test with realistic scheduler data
test_cases = [
    ([32, 16, 16, 0.8, 0.3, 0, 0, 0, 0, 0], "High load imbalance"),
    ([64, 32, 32, 0.5, 0.5, 0, 0, 0, 0, 0], "Balanced load"),
    ([16, 48, 16, 0.2, 0.2, 0, 0, 0, 0, 0], "Low load system"),
]

print("\nTest predictions:")
for features, desc in test_cases:
    test_input = tf.constant([features], dtype=tf.float32)
    pred = wrapper_model.predict(test_input, verbose=0)[0][0]
    print(f"{desc:20} -> {pred:.4f} -> {'MIGRATE' if pred > 0.5 else 'STAY'}")

print("\n✓ Model ready for use with ML scheduler!")