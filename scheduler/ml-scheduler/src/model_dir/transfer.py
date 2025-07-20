#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras

# Load the Keras model
model = keras.models.load_model('model_stress_model.h5')

# Export to SavedModel format
tf.saved_model.save(model, 'model_path')