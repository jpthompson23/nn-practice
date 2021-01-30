import numpy as np


class NpNet(object):
    def __init__(self, w1: np.ndarray = None, w2: np.ndarray = None):
        print("Initializing NpNet...")
        self.w1 = np.zeros((28*28, 128), dtype=np.float32)
        if w1 is not None:
            self.w1[:] = w1
        self.w2 = np.zeros((128, 10), dtype=np.float32)
        if w2 is not None:
            self.w2[:] = w2

    def forward(self, x):
        x = x.dot(self.w1)
        # rectified linear:
        x = np.maximum(x, 0.0)
        x = x.dot(self.w2)
        return x

    def predict(self, x: np.ndarray):
        y_test_preds = np.argmax(self.forward(x.reshape(-1, 28*28)), axis=1)
        return y_test_preds
