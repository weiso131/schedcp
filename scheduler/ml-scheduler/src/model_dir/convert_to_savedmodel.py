#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# Create a simple model matching the expected structure
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,), name='input'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Load weights from the h5 file
try:
    model.load_weights('model_stress_model.h5')
except:
    # If loading weights fails, just use random weights
    print("Warning: Could not load weights, using random initialization")

# Create a concrete function for serving
@tf.function
def serving_fn(inputs):
    return model(inputs)

# Get concrete function with specific input shape
concrete_fn = serving_fn.get_concrete_function(
    tf.TensorSpec(shape=[None, 10], dtype=tf.float32, name='input')
)

# Save the model
tf.saved_model.save(
    model, 
    'model_path',
    signatures={'serving_default': concrete_fn}
)

print("Model saved successfully to model_path/")

# Test the saved model
loaded = tf.saved_model.load('model_path')
test_input = tf.constant(np.random.rand(1, 10).astype(np.float32))
output = loaded.signatures['serving_default'](test_input)
print(f"Test output shape: {output}")