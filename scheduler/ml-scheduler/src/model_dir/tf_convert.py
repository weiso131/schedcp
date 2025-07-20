#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the model
print("Loading model...")
model = tf.keras.models.load_model('model_stress_model.h5')

# Convert to TensorFlow Lite as a workaround
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted to TFLite format: model.tflite")

# For now, let's use the .h5 file directly in our Rust code
print("Note: Using .h5 file directly for TensorFlow Rust bindings")