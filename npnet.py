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
        y_test_preds = self.forward(x.reshape(-1, 28*28))
        return y_test_preds

    def loss(self, y_preds: np.ndarray, y_test: np.ndarray):
        print("y_preds.shape: ", y_preds.shape)
        print("y_preds: ", y_preds)

        print("y_test.shape", y_test.shape)
        print("y_test: ", y_test)

        y_test_reshaped = y_test.reshape((y_test.shape[0], 1))
        print("y_test_reshaped: ", y_test_reshaped)
        scores = np.take_along_axis(y_preds, y_test_reshaped, axis=1)
        scores = scores.reshape(y_test.shape)
        print("scores: ", scores)
        return -scores + np.log(np.exp(y_preds).sum(axis=1))
