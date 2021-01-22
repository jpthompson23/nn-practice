import numpy as np


class NpNet(object):
    def __init__(self):
        self.w1 = np.zeros((28*28, 128), dtype=np.float32)
        self.w2 = np.zeros((128, 10), dtype=np.float32)

    def forward(self, x):
        # numpy forward pass
        pass
