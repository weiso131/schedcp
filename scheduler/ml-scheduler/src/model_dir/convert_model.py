#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
import tensorflow as tf
from tensorflow import keras

# Load the Keras model
model = keras.models.load_model('model_stress_model.h5')

# Create a concrete function for the model
@tf.function
def serving_default(inputs):
    return model(inputs, training=False)

# Get the concrete function with specified input shape
concrete_func = serving_default.get_concrete_function(
    tf.TensorSpec(shape=[None, 5], dtype=tf.float32, name='serving_default_input')
)

# Save the model using the lower-level API
tf.saved_model.save(
    model, 
    'model_path',
    signatures={'serving_default': concrete_func}
)

print("Model successfully converted to SavedModel format at 'model_path'")

# Verify the saved model
loaded = tf.saved_model.load('model_path')
print("Model loaded successfully!")
print("Available signatures:", list(loaded.signatures.keys()))