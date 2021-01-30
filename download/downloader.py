import requests
import os

DATA_DIR = "./data"

URLS = {
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz": "x-train.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz": "y-train.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz": "x-test.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz": "y-test.gz"
}


def fetch(url, filename, data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    filepath = os.path.join(data_dir, filename)
    if not os.path.isfile(filepath):
        with open(filepath, 'wb') as f:
            content = requests.get(url).content
            f.write(content)


if __name__ == "__main__":
    for url, filename in URLS.items():
        fetch(url, filename, data_dir=DATA_DIR)
