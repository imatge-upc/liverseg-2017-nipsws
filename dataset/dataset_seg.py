"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)
Class to define the Dataset object.
"""

from PIL import Image
import os
import numpy as np
import scipy.io


class Dataset:
    def __init__(self, train_list, test_list, val_list, database_root, number_of_slices, store_memory=True):
        """Initialize the Dataset object
        Args:
        train_list: TXT file with the path to the images to use for training (Images must be between 0 and 255)
        test_list: TXT file with the path to the images to use for testing (Images must be between 0 and 255)
        database_root: Path to the root of the Database
        store_memory: True stores all the training images, False loads at runtime the images
        Returns:
        """
        # Load training images (path) and labels
        print('Started loading files...')
        if train_list is not None:
            with open(train_list) as t:
                train_paths = t.readlines()
        else:
            train_paths = []
        if test_list is not None:
            with open(test_list) as t:
                test_paths = t.readlines()
        else:
            test_paths = []

        if val_list is not None:
            with open(val_list) as t:
                val_paths = t.readlines()
        else:
            val_paths = []

        self.images_train = []
        self.images_train_path = []
        self.labels_train = []
        self.labels_train_path = []
        self.labels_liver_train = []
        self.labels_liver_train_path = []
        for idx, line in enumerate(train_paths):
            if (len(line.split()) > 3):
                if store_memory:
                    aux_images_train = []
                    aux_labels_train = []
                    aux_labels_liver_train = []
                    for i in range(number_of_slices):
                        aux_images_train.append(
                            np.array(scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3])))['section'],
                                     dtype=np.float32))
                    self.images_train.append(np.array(aux_images_train))

                    for i in range(number_of_slices):
                        aux_labels_train.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 1])))['section'],
                            dtype=np.float32))
                    self.labels_train.append(np.array(aux_labels_train))

                    for i in range(number_of_slices):
                        aux_labels_liver_train.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 2])))['section'],
                            dtype=np.float32))
                    self.labels_liver_train.append(np.array(aux_labels_liver_train))

                    if (idx + 1) % 1000 == 0:
                        print('Loaded ' + str(idx) + ' train images')

                aux_images_train_path = []
                aux_labels_train_path = []
                aux_labels_liver_train_path = []

                for i in range(number_of_slices):
                    aux_images_train_path.append(os.path.join(database_root, str(line.split()[i * 3])))
                self.images_train_path.append(np.array(aux_images_train_path))

                for i in range(number_of_slices):
                    aux_labels_train_path.append(os.path.join(database_root, str(line.split()[i * 3 + 1])))
                self.labels_train_path.append(np.array(aux_labels_train_path))

                for i in range(number_of_slices):
                    aux_labels_liver_train_path.append(os.path.join(database_root, str(line.split()[i * 3 + 2])))
                self.labels_liver_train_path.append(np.array(aux_labels_liver_train_path))

        self.images_train_path = np.array(self.images_train_path)
        self.labels_train_path = np.array(self.labels_train_path)
        self.labels_liver_train_path = np.array(self.labels_liver_train_path)

        # Load testing images (path) and labels
        self.images_test = []
        self.images_test_path = []
        for idx, line in enumerate(test_paths):
            if (len(line.split()) > 1):
                if store_memory:
                    aux_images_test = []
                    for i in range(number_of_slices):
                        aux_images_test.append(
                            np.array(scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3])))['section'],
                                     dtype=np.float32))
                    self.images_test.append(np.array(aux_images_test))

                    if (idx + 1) % 1000 == 0:
                        print('Loaded ' + str(idx) + ' test images')

                aux_images_test_path = []
                for i in range(number_of_slices):
                    aux_images_test_path.append(os.path.join(database_root, str(line.split()[i * 3])))
                self.images_test_path.append(np.array(aux_images_test_path))

        self.images_val = []
        self.images_val_path = []
        self.labels_val = []
        self.labels_val_path = []
        self.labels_liver_val = []
        self.labels_liver_val_path = []
        for idx, line in enumerate(val_paths):
            if (len(line.split()) > 3):
                if store_memory:
                    aux_images_val = []
                    aux_labels_val = []
                    aux_labels_liver_val = []
                    for i in range(number_of_slices):
                        aux_images_val.append(
                            np.array(scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3])))['section'],
                                     dtype=np.float32))
                    self.images_val.append(np.array(aux_images_val))

                    for i in range(number_of_slices):
                        aux_labels_val.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 1])))['section'],
                            dtype=np.float32))
                    self.labels_val.append(np.array(aux_labels_val))

                    for i in range(number_of_slices):
                        aux_labels_liver_val.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 2])))['section'],
                            dtype=np.float32))
                    self.labels_liver_val.append(np.array(aux_labels_liver_val))

                    if (idx + 1) % 1000 == 0:
                        print('Loaded ' + str(idx) + ' train images')

                aux_images_val_path = []
                aux_labels_val_path = []
                aux_labels_liver_val_path = []

                for i in range(number_of_slices):
                    aux_images_val_path.append(os.path.join(database_root, str(line.split()[i * 3])))
                self.images_val_path.append(np.array(aux_images_val_path))

                for i in range(number_of_slices):
                    aux_labels_val_path.append(os.path.join(database_root, str(line.split()[i * 3 + 1])))
                self.labels_val_path.append(np.array(aux_labels_val_path))

                for i in range(number_of_slices):
                    aux_labels_liver_val_path.append(os.path.join(database_root, str(line.split()[i * 3 + 2])))
                self.labels_liver_val_path.append(np.array(aux_labels_liver_val_path))

        self.images_val_path = np.array(self.images_val_path)
        self.labels_val_path = np.array(self.labels_val_path)
        self.labels_liver_val_path = np.array(self.labels_liver_val_path)

        print('Done initializing Dataset')

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.val_ptr = 0
        self.train_size = len(self.images_train_path)
        self.test_size = len(self.images_test_path)
        self.val_size = len(self.images_val_path)
        self.train_idx = np.arange(self.train_size)
        self.val_idx = np.arange(self.val_size)
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.val_idx)
        self.store_memory = store_memory

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train' or 'test'
        Returns in training:
        images: List of images paths if store_memory=False, List of Numpy arrays of the images if store_memory=True
        labels: List of labels paths if store_memory=False, List of Numpy arrays of the labels if store_memory=True
        Returns in testing:
        images: None if store_memory=False, Numpy array of the image if store_memory=True
        path: List of image paths
        """
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                if self.store_memory:
                    images = [self.images_train[l] for l in idx]
                    labels = [self.labels_train[l] for l in idx]
                    labels_liver = [self.labels_liver[l] for l in idx]
                else:
                    images = [self.images_train_path[l] for l in idx]
                    labels = [self.labels_train_path[l] for l in idx]
                    labels_liver = [self.labels_liver_train_path[l] for l in idx]
                self.train_ptr += batch_size
            else:
                old_idx = np.array(self.train_idx[self.train_ptr:])
                np.random.shuffle(self.train_idx)
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                idx = np.array(self.train_idx[:new_ptr])
                if self.store_memory:
                    images_1 = [self.images_train[l] for l in old_idx]
                    labels_1 = [self.labels_train[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_train[l] for l in old_idx]
                    images_2 = [self.images_train[l] for l in idx]
                    labels_2 = [self.labels_train[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path[l] for l in old_idx]
                    labels_1 = [self.labels_train_path[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_train_path[l] for l in old_idx]
                    images_2 = [self.images_train_path[l] for l in idx]
                    labels_2 = [self.labels_train_path[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_train_path[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                labels_liver = labels_liver_1 + labels_liver_2
                self.train_ptr = new_ptr
            return images, labels, labels_liver
        if phase == 'val':
            if self.val_ptr + batch_size < self.val_size:
                idx = np.array(self.val_idx[self.val_ptr:self.val_ptr + batch_size])
                if self.store_memory:
                    images = [self.images_val[l] for l in idx]
                    labels = [self.labels_val[l] for l in idx]
                    labels_liver = [self.labels_liver_val[l] for l in idx]
                else:
                    images = [self.images_val_path[l] for l in idx]
                    labels = [self.labels_val_path[l] for l in idx]
                    labels_liver = [self.labels_liver_val_path[l] for l in idx]
                self.val_ptr += batch_size
            else:
                old_idx = np.array(self.val_idx[self.val_ptr:])
                np.random.shuffle(self.val_idx)
                new_ptr = (self.val_ptr + batch_size) % self.val_size
                idx = np.array(self.val_idx[:new_ptr])
                if self.store_memory:
                    images_1 = [self.images_val[l] for l in old_idx]
                    labels_1 = [self.labels_val[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_val[l] for l in old_idx]
                    images_2 = [self.images_val[l] for l in idx]
                    labels_2 = [self.labels_val[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_val[l] for l in idx]
                else:
                    images_1 = [self.images_val_path[l] for l in old_idx]
                    labels_1 = [self.labels_val_path[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_val_path[l] for l in old_idx]
                    images_2 = [self.images_val_path[l] for l in idx]
                    labels_2 = [self.labels_val_path[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_val_path[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                labels_liver = labels_liver_1 + labels_liver_2
                self.val_ptr = new_ptr
            return images, labels, labels_liver
        elif phase == 'test':
            images = None
            if self.test_ptr + batch_size < self.test_size:
                if self.store_memory:
                    images = self.images_test[self.test_ptr:self.test_ptr + batch_size]
                paths = self.images_test_path[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                if self.store_memory:
                    images = self.images_test[self.test_ptr:] + self.images_test[:new_ptr]
                paths = self.images_test_path[self.test_ptr:] + self.images_test_path[:new_ptr]
                self.test_ptr = new_ptr
            return images, paths
        else:
            return None, None

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def get_val_size(self):
        return self.val_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width

        if train_list is not None:
            with open(train_list) as t:
                train_paths = t.readlines()
        else:
            train_paths = []
        if test_list is not None:
            with open(test_list) as t:
                test_paths = t.readlines()
        else:
            test_paths = []

        if val_list is not None:
            with open(val_list) as t:
                val_paths = t.readlines()
        else:
            val_paths = []

        self.images_train = []
        self.images_train_path = []
        self.labels_train = []
        self.labels_train_path = []
        self.labels_liver_train = []
        self.labels_liver_train_path = []
        for idx, line in enumerate(train_paths):
            if (len(line.split()) > 3):
                if store_memory:
                    aux_images_train = []
                    aux_labels_train = []
                    aux_labels_liver_train = []
                    for i in range(number_of_slices):
                        aux_images_train.append(
                            np.array(scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3])))['section'],
                                     dtype=np.float32))
                    self.images_train.append(np.array(aux_images_train))

                    for i in range(number_of_slices):
                        aux_labels_train.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 1])))['section'],
                            dtype=np.float32))
                    self.labels_train.append(np.array(aux_labels_train))

                    for i in range(number_of_slices):
                        aux_labels_liver_train.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 2])))['section'],
                            dtype=np.float32))
                    self.labels_liver_train.append(np.array(aux_labels_liver_train))

                    if (idx + 1) % 1000 == 0:
                        print('Loaded ' + str(idx) + ' train images')

                aux_images_train_path = []
                aux_labels_train_path = []
                aux_labels_liver_train_path = []

                for i in range(number_of_slices):
                    aux_images_train_path.append(os.path.join(database_root, str(line.split()[i * 3])))
                self.images_train_path.append(np.array(aux_images_train_path))

                for i in range(number_of_slices):
                    aux_labels_train_path.append(os.path.join(database_root, str(line.split()[i * 3 + 1])))
                self.labels_train_path.append(np.array(aux_labels_train_path))

                for i in range(number_of_slices):
                    aux_labels_liver_train_path.append(os.path.join(database_root, str(line.split()[i * 3 + 2])))
                self.labels_liver_train_path.append(np.array(aux_labels_liver_train_path))

        self.images_train_path = np.array(self.images_train_path)
        self.labels_train_path = np.array(self.labels_train_path)
        self.labels_liver_train_path = np.array(self.labels_liver_train_path)

        # Load testing images (path) and labels
        self.images_test = []
        self.images_test_path = []
        for idx, line in enumerate(test_paths):
            if (len(line.split()) > 1):
                if store_memory:
                    aux_images_test = []
                    for i in range(number_of_slices):
                        aux_images_test.append(
                            np.array(scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3])))['section'],
                                     dtype=np.float32))
                    self.images_test.append(np.array(aux_images_test))

                    if (idx + 1) % 1000 == 0:
                        print('Loaded ' + str(idx) + ' test images')

                aux_images_test_path = []
                for i in range(number_of_slices):
                    aux_images_test_path.append(os.path.join(database_root, str(line.split()[i * 3])))
                self.images_test_path.append(np.array(aux_images_test_path))

        self.images_val = []
        self.images_val_path = []
        self.labels_val = []
        self.labels_val_path = []
        self.labels_liver_val = []
        self.labels_liver_val_path = []
        for idx, line in enumerate(val_paths):
            if (len(line.split()) > 3):
                if store_memory:
                    aux_images_val = []
                    aux_labels_val = []
                    aux_labels_liver_val = []
                    for i in range(number_of_slices):
                        aux_images_val.append(
                            np.array(scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3])))['section'],
                                     dtype=np.float32))
                    self.images_val.append(np.array(aux_images_val))

                    for i in range(number_of_slices):
                        aux_labels_val.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 1])))['section'],
                            dtype=np.float32))
                    self.labels_val.append(np.array(aux_labels_val))

                    for i in range(number_of_slices):
                        aux_labels_liver_val.append(np.array(
                            scipy.io.loadmat(os.path.join(database_root, str(line.split()[i * 3 + 2])))['section'],
                            dtype=np.float32))
                    self.labels_liver_val.append(np.array(aux_labels_liver_val))

                    if (idx + 1) % 1000 == 0:
                        print('Loaded ' + str(idx) + ' train images')

                aux_images_val_path = []
                aux_labels_val_path = []
                aux_labels_liver_val_path = []

                for i in range(number_of_slices):
                    aux_images_val_path.append(os.path.join(database_root, str(line.split()[i * 3])))
                self.images_val_path.append(np.array(aux_images_val_path))

                for i in range(number_of_slices):
                    aux_labels_val_path.append(os.path.join(database_root, str(line.split()[i * 3 + 1])))
                self.labels_val_path.append(np.array(aux_labels_val_path))

                for i in range(number_of_slices):
                    aux_labels_liver_val_path.append(os.path.join(database_root, str(line.split()[i * 3 + 2])))
                self.labels_liver_val_path.append(np.array(aux_labels_liver_val_path))

        self.images_val_path = np.array(self.images_val_path)
        self.labels_val_path = np.array(self.labels_val_path)
        self.labels_liver_val_path = np.array(self.labels_liver_val_path)

        print('Done initializing Dataset')

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.val_ptr = 0
        self.train_size = len(self.images_train_path)
        self.test_size = len(self.images_test_path)
        self.val_size = len(self.images_val_path)
        self.train_idx = np.arange(self.train_size)
        self.val_idx = np.arange(self.val_size)
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.val_idx)
        self.store_memory = store_memory

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train' or 'test'
        Returns in training:
        images: List of images paths if store_memory=False, List of Numpy arrays of the images if store_memory=True
        labels: List of labels paths if store_memory=False, List of Numpy arrays of the labels if store_memory=True
        Returns in testing:
        images: None if store_memory=False, Numpy array of the image if store_memory=True
        path: List of image paths
        """
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                if self.store_memory:
                    images = [self.images_train[l] for l in idx]
                    labels = [self.labels_train[l] for l in idx]
                    labels_liver = [self.labels_liver[l] for l in idx]
                else:
                    images = [self.images_train_path[l] for l in idx]
                    labels = [self.labels_train_path[l] for l in idx]
                    labels_liver = [self.labels_liver_train_path[l] for l in idx]
                self.train_ptr += batch_size
            else:
                old_idx = np.array(self.train_idx[self.train_ptr:])
                np.random.shuffle(self.train_idx)
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                idx = np.array(self.train_idx[:new_ptr])
                if self.store_memory:
                    images_1 = [self.images_train[l] for l in old_idx]
                    labels_1 = [self.labels_train[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_train[l] for l in old_idx]
                    images_2 = [self.images_train[l] for l in idx]
                    labels_2 = [self.labels_train[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path[l] for l in old_idx]
                    labels_1 = [self.labels_train_path[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_train_path[l] for l in old_idx]
                    images_2 = [self.images_train_path[l] for l in idx]
                    labels_2 = [self.labels_train_path[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_train_path[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                labels_liver = labels_liver_1 + labels_liver_2
                self.train_ptr = new_ptr
            return images, labels, labels_liver
        if phase == 'val':
            if self.val_ptr + batch_size < self.val_size:
                idx = np.array(self.val_idx[self.val_ptr:self.val_ptr + batch_size])
                if self.store_memory:
                    images = [self.images_val[l] for l in idx]
                    labels = [self.labels_val[l] for l in idx]
                    labels_liver = [self.labels_liver_val[l] for l in idx]
                else:
                    images = [self.images_val_path[l] for l in idx]
                    labels = [self.labels_val_path[l] for l in idx]
                    labels_liver = [self.labels_liver_val_path[l] for l in idx]
                self.val_ptr += batch_size
            else:
                old_idx = np.array(self.val_idx[self.val_ptr:])
                np.random.shuffle(self.val_idx)
                new_ptr = (self.val_ptr + batch_size) % self.val_size
                idx = np.array(self.val_idx[:new_ptr])
                if self.store_memory:
                    images_1 = [self.images_val[l] for l in old_idx]
                    labels_1 = [self.labels_val[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_val[l] for l in old_idx]
                    images_2 = [self.images_val[l] for l in idx]
                    labels_2 = [self.labels_val[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_val[l] for l in idx]
                else:
                    images_1 = [self.images_val_path[l] for l in old_idx]
                    labels_1 = [self.labels_val_path[l] for l in old_idx]
                    labels_liver_1 = [self.labels_liver_val_path[l] for l in old_idx]
                    images_2 = [self.images_val_path[l] for l in idx]
                    labels_2 = [self.labels_val_path[l] for l in idx]
                    labels_liver_2 = [self.labels_liver_val_path[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                labels_liver = labels_liver_1 + labels_liver_2
                self.val_ptr = new_ptr
            return images, labels, labels_liver
        elif phase == 'test':
            images = None
            if self.test_ptr + batch_size < self.test_size:
                if self.store_memory:
                    images = self.images_test[self.test_ptr:self.test_ptr + batch_size]
                paths = self.images_test_path[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                if self.store_memory:
                    images = self.images_test[self.test_ptr:] + self.images_test[:new_ptr]
                paths = self.images_test_path[self.test_ptr:] + self.images_test_path[:new_ptr]
                self.test_ptr = new_ptr
            return images, paths
        else:
            return None, None

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def get_val_size(self):
        return self.val_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width
