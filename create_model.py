import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf

def main():
    start = 0
    end = 100000

    range_values = np.arange(start, end)

    # Create the digit representation as a 2D array
    X = np.array([list(map(int, str(x).zfill(15))) for x in range_values])

    # Create the labels using vectorized operations
    y = np.column_stack((range_values % 2, (range_values + 1) % 2))

    # Split data up into training and validation data.
    split_train = StratifiedShuffleSplit(n_splits=3, test_size=0.4, train_size=0.6)
    for train_index, test_index in split_train.split(X, y):
        X_val, X_train = X[test_index], X[train_index]
        y_val, y_train = y[test_index], y[train_index]


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(70, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=512, epochs=500, validation_data=(X_val, y_val))

    print(f"maximum accurary of '{max(model.history.history['val_accuracy'])}' at epoch number {model.history.history['val_accuracy'].index(max(model.history.history['val_accuracy']))}")
    print(f"mimimum loss of '{min(model.history.history['val_loss'])}' at epoch number {model.history.history['val_loss'].index(min(model.history.history['val_loss']))}")

    model.save('model.keras')


if __name__ == '__main__':
    main()