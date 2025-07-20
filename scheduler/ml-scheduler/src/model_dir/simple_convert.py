#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Load and convert model
model = tf.keras.models.load_model('model_stress_model.h5')

# Save in SavedModel format
model.save('model_path', save_format='tf')
print("Model converted successfully to model_path/")