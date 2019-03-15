import numpy as np


class DataSet(object):

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._one_hot_labels = to_one_hot_encoding(labels)
        self._features = None
        self._num_samples = self._images.shape[0]
        self._num_epochs = 0
        self._index = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def one_hot_labels(self):
        return self._one_hot_labels

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_epochs(self):
        return self._num_epochs

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples, features and labels from this data set."""
        assert batch_size <= self._num_samples

        if self._index + batch_size >= self._num_samples:
            perm = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._features = self._features[perm]
            self._index = 0
            self._num_epochs += 1

        images = self._images[self._index: self._index + batch_size]
        labels = self._labels[self._index: self._index + batch_size]
        features = self._features[self._index: self._index + batch_size]
        self._index += batch_size
        return images, features, labels, to_one_hot_encoding(labels)


class DataSets(object):

    def __init__(self, train, valid, test):
        self._train = train
        self._test = test
        self._valid = valid

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test


def to_one_hot_encoding(labels):
    num_classes = np.max(labels) + 1
    one_hot_labels = np.zeros(shape=(len(labels), num_classes), dtype=np.float32)
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1.0
    return one_hot_labels
