import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.keras')

model.predict(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))