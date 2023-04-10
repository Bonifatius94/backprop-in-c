from timeit import timeit
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from keras import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam


def main():
    num_samples = 10_000
    x = np.expand_dims(np.linspace(-10, 10, num_samples), axis=-1)
    y = np.sin(x)

    perm = np.arange(num_samples)
    np.random.shuffle(perm)
    x, y = x[perm], y[perm]

    def train(x: np.ndarray, y: np.ndarray) -> Model:
        feature_dims, label_dims = x.shape[1], y.shape[1]
        model = model = Sequential([
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(label_dims, activation="linear"),
        ])
        model.build((None, feature_dims))
        model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
        model.fit(x, y, epochs=10, batch_size=64, validation_split=0.2)
        return model

    repetitions = 10
    time_secs = timeit(lambda: train(x, y), number=repetitions)
    print(f"time per training: {(time_secs / repetitions):.2f} s")


if __name__ == "__main__":
    main()
