import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles


class ClassObj:

    def __init__(self, _class, data_dim, size):
        self._class = _class
        self.data_dim = data_dim
        self.size = size
        self.data_points = np.array([])

    def output_data(self):
        y = (np.repeat(self._class, self.size).astype(np.float32)[:, np.newaxis])
        return np.concatenate((self.data_points, y), axis=1)


class Circular(ClassObj):

    def __init__(self, _class, data_dim, ray, noise, size, seed):
        super(Circular, self).__init__(_class, data_dim, size)
        self.ray = ray
        self.noise = noise
        self.seed = seed

    def generate_data(self):
        self.data_points = (make_circles(n_samples=(self.size, 0), noise=self.noise, random_state=self.seed)[0]
                            * self.ray).astype(np.float32)


class Dataset:

    def __init__(self, classes=None):
        self.class_container = classes or []

    def add_class(self, class_):
        self.class_container.append(class_)

    def generate_classes(self):
        for class_ in self.class_container:
            class_.generate_data()

    def output_data(self):
        data = np.vstack([class_.output_data() for class_ in self.class_container])
        X, y = data[:, :-1], data[:, -1][:, np.newaxis]
        return X, y

    def plot_data(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        X = self.output_data()[0]
        y_fine = self.output_data()[1]
        y_coarse = self.output_data()[2]

        ax1.scatter(X[:, 0], X[:, 1], s=10, c=y_fine)

        ax2.scatter(X[:, 0], X[:, 1], s=10, c=y_coarse)

        return fig, ax1, ax2
