import numpy as np


class DataSet(object):

    def __init__(self, images, labels, positive_images, negative_images):
        self._images = images
        self._labels = labels
        self._positive_images = positive_images
        self._negative_images = negative_images
        self._num_positive_images = positive_images.shape[0]
        self._num_negative_images = negative_images.shape[0]
        self._index_in_positive_images = 0
        self._index_in_negative_images = 0
        self._num_positive_epochs = 0
        self._num_negative_epochs = 0
        self._positive_image_features = None
        self._negative_image_features = None

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def positive_images(self):
        return self._positive_images

    @property
    def negative_images(self):
        return self._negative_images

    @property
    def positive_image_features(self):
        return self._positive_image_features

    @positive_image_features.setter
    def positive_image_features(self, value):
        self._positive_image_features = value

    @property
    def negative_image_features(self):
        return self._negative_image_features

    @negative_image_features.setter
    def negative_image_features(self, value):
        self._negative_image_features = value

    @property
    def num_epochs(self):
        return max(self._num_positive_epochs, self._num_negative_epochs)

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples, features and labels from this data set."""
        batch_size //= 2
        start_positive_images = self._index_in_positive_images
        self._index_in_positive_images += batch_size
        if self._index_in_positive_images > self._num_positive_images:
            # Shuffle the data
            perm = np.arange(self._num_positive_images)
            np.random.shuffle(perm)
            self._positive_images = self._positive_images[perm]
            self._positive_image_features = self._positive_image_features[perm]
            # Start next epoch
            start_positive_images = 0
            self._index_in_positive_images = batch_size
            assert batch_size <= self._num_positive_images
            self._num_positive_epochs += 1
        end_positive_images = self._index_in_positive_images

        start_negative_images = self._index_in_negative_images
        self._index_in_negative_images += batch_size
        if self._index_in_negative_images > self._num_negative_images:
            # Shuffle the data
            perm = np.arange(self._num_negative_images)
            np.random.shuffle(perm)
            self._negative_images = self._negative_images[perm]
            self._negative_image_features = self._negative_image_features[perm]
            # Start next epoch
            start_negative_images = 0
            self._index_in_negative_images = batch_size
            assert batch_size <= self._num_negative_images
            self._num_negative_epochs += 1
        end_negative_images = self._index_in_negative_images

        images = np.vstack((self._positive_images[start_positive_images: end_positive_images],
                            self._negative_images[start_negative_images: end_negative_images]))
        features = np.vstack((self._positive_image_features[start_positive_images: end_positive_images],
                              self._negative_image_features[start_negative_images: end_negative_images]))
        labels = np.vstack((np.ones(shape=(batch_size, 1), dtype=np.float32),
                            np.zeros(shape=(batch_size, 1), dtype=np.float32)))
        return images, features, labels

    def balanced_all(self):
        positive_multiplier = int(max(self._num_negative_images/self._num_positive_images, 1.0))
        negative_multiplier = int(max(self._num_positive_images/self._num_negative_images, 1.0))
        images = np.vstack((np.vstack([self._positive_images for _ in range(positive_multiplier)]),
                            np.vstack([self._negative_images for _ in range(negative_multiplier)])))
        features = np.vstack((np.vstack([self._positive_image_features for _ in range(positive_multiplier)]),
                              np.vstack([self._negative_image_features for _ in range(negative_multiplier)])))
        labels = np.vstack((np.ones(shape=(self._num_positive_images * positive_multiplier, 1), dtype=np.float32),
                            np.zeros(shape=(self._num_negative_images * negative_multiplier, 1), dtype=np.float32)))
        return images, features, labels


class DataSets(object):

    def __init__(self, train, test):
        self._train = train
        self._test = test

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
