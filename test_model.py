import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.keras')

def num_to_nparray(num):
    return np.array([list(map(int, str(x).zfill(15)))])

def predict_odd(num):
    prediction = model.predict(num_to_nparray(num))
    return prediction[0][0] > prediction[0][1]

for i in range(1_000_000_000_000_000):
    predict = predict_odd(i)
    correct = bool(i%2)
    if( predict != correct):
        print(f"error at number {i}, should be {correct}, predicted {predict}")
        break