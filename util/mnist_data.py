import os
import urllib.request
import gzip
import numpy as np
from util.DataSet import DataSet, DataSets


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images_and_labels(image_file, label_file, positive_label, percentage=1.0):
    """Extract the images into two 4D uint8 numpy array [index, y, x, depth]: positive and negative images."""
    print('Extracting', image_file, label_file)
    with gzip.open(image_file) as image_bytestream, gzip.open(label_file) as label_bytestream:
        magic = _read32(image_bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in image file: %s' %
                (magic, image_file))
        magic = _read32(label_bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in label file: %s' %
                (magic, label_file))
        num_images = _read32(image_bytestream)
        rows = _read32(image_bytestream)
        cols = _read32(image_bytestream)
        num_labels = _read32(label_bytestream)
        if num_images != num_labels:
            raise ValueError(
                'Num images does not match num labels. Image file : %s; label file: %s' %
                (image_file, label_file))
        positive_images = []
        negative_images = []
        images = []
        labels = []
        num_images = int(num_images * percentage)
        for _ in range(num_images):
            image_buf = image_bytestream.read(rows * cols)
            image = np.frombuffer(image_buf, dtype=np.uint8)
            image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
            image[np.where(image == 0.0)[0]] = 1e-7
            image[np.where(image == 1.0)[0]] -= 1e-7
            label = np.frombuffer(label_bytestream.read(1), dtype=np.uint8)
            if label[0] == positive_label:
                positive_images.append(image)
            else:
                negative_images.append(image)
            images.append(image)
            labels.append(label)
        positive_images = np.array(positive_images, dtype=np.float32)
        negative_images = np.array(negative_images, dtype=np.float32)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)
        return images, labels, positive_images, negative_images


def read_data_sets(train_dir, positive_label, percentage=1.0):
    train_image_file = 'train-images-idx3-ubyte.gz'
    train_label_file = 'train-labels-idx1-ubyte.gz'
    test_image_file = 't10k-images-idx3-ubyte.gz'
    test_label_file = 't10k-labels-idx1-ubyte.gz'

    train_image_file = maybe_download(train_image_file, train_dir)
    train_label_file = maybe_download(train_label_file, train_dir)
    train_images, train_labels, train_positive_images, train_negative_images = \
        extract_images_and_labels(train_image_file, train_label_file, positive_label, percentage)
    test_image_file = maybe_download(test_image_file, train_dir)
    test_label_file = maybe_download(test_label_file, train_dir)
    test_images, test_labels, test_positive_images, test_negative_images = \
        extract_images_and_labels(test_image_file, test_label_file, positive_label)

    train = DataSet(train_images, train_labels, train_positive_images, train_negative_images)
    test = DataSet(test_images, test_labels, test_positive_images, test_negative_images)
    return DataSets(train, test)
