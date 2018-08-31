import numpy as np
import pandas as pd
from tensorflow.python.framework import random_seed


def read_macro_batch(index, sample_size, shuffle=False):
    data = pd.read_csv('/home/yuanning/python/deeprep/data/train/Train_data_normed_batch{}.dat'.format(index),
                       delimiter=',', header=None)
    # print('loading batch file {}.dat'.format(index))
    data = data.values
    label = pd.read_csv('/home/yuanning/python/deeprep/data/train/Train_data_label_batch{}.dat'.format(index),
                        delimiter=',', header=None)
    label = label.values
    data3d = array_2d_to_3d(data, sample_size)
    if shuffle:
        perm0 = np.arange(data3d.shape[0])
        np.random.shuffle(perm0)
        data3d = data3d[perm0]
        label = label[perm0]
    return data3d, label


def read_macro_batch_FBL(index, sample_size, shuffle=False):
    data = pd.read_csv('/home/yuanning/python/deeprep/data/test/Test_data_normed_FBL_batch{}.dat'.format(index),
                       delimiter=',', header=None)
    data = data.values
    label = pd.read_csv('/home/yuanning/python/deeprep/data/test/Test_data_FBL_label_batch{}.dat'.format(index),
                        delimiter=',', header=None)
    label = label.values
    data3d = array_2d_to_3d(data, sample_size)
    if shuffle:
        perm0 = np.arange(data3d.shape[0])
        np.random.shuffle(perm0)
        data3d = data3d[perm0]
        label = label[perm0]
    return data3d, label


def read_macro_batch_FWL(index, sample_size, shuffle=False):
    data = pd.read_csv('/home/yuanning/python/deeprep/data/test/Test_data_normed_FWL_batch{}.dat'.format(index),
                       delimiter=',', header=None)
    data = data.values
    label = pd.read_csv('/home/yuanning/python/deeprep/data/test/Test_data_FWL_label_batch{}.dat'.format(index),
                        delimiter=',', header=None)
    label = label.values
    data3d = array_2d_to_3d(data, sample_size)
    if shuffle:
        perm0 = np.arange(data3d.shape[0])
        np.random.shuffle(perm0)
        data3d = data3d[perm0]
        label = label[perm0]
    return data3d, label


def trunc_data(data3d, index):
    return data3d[:, :, index]


def array_3d_to_2d(data_3d):
    n = data_3d.shape[0]
    p1 = data_3d.shape[1]
    p2 = data_3d.shape[2]
    data_2d = np.zeros((n, p1*p2))
    for i in range(data_3d.shape[0]):
        data_2d[i, :] = np.reshape(data_3d[i, :, :], (p1*p2, ))
    return data_2d


def array_2d_to_3d(data_2d, sample_size):
    n = data_2d.shape[0]
    p = data_2d.shape[1]
    num_trials = n/sample_size
    data_3d = np.zeros((num_trials, sample_size, p))
    for i in range(num_trials):
        data_3d[i, :, :] = data_2d[(i*sample_size):((i+1)*sample_size), :]
    return data_3d


class DataSet(object):
    def __init__(self, num_macro_batches, num_trials_in_macro_batch, sample_size, data_type, shuffle=True):
        self._epochs_completed = 0            # number of epochs completed
        self._macro_batch_index_in_epoch = 0  # index of the current macro batch
        self._index_in_macro_batch = 0        # index of the current sample
        self._num_train_macro_batches = num_macro_batches
        self._num_trials_in_macro_batch = num_trials_in_macro_batch
        self._sample_size = sample_size  # length of each sample e.g. 100 (200 ms)
        self._batch_list = np.arange(num_macro_batches)  # list of the macro batches to loop
        data, label = read_macro_batch(0, sample_size, shuffle)
        self._current_macro_batch = data  # current macro batch data
        self._current_label = label       # current macro batch label
        self._data_type = data_type       # train, FBL or FWL

    @property
    def current_macro_batch(self):
        return self._current_macro_batch

    @property
    def current_label(self):
        return self._current_label

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def num_macro_batches(self):
        return self._num_train_macro_batches

    @property
    def size_of_macro_batch(self):
        return self._num_trials_in_macro_batch

    def next_batch(self, batch_size, shuffle=True):
        start_point_macro_batch = self._macro_batch_index_in_epoch
        index_in_macro_batch = self._index_in_macro_batch
        # shuffle in the first epoch
        if self._epochs_completed == 0 and start_point_macro_batch == 0 and index_in_macro_batch == 0:
            perm_batch = np.arange(self._num_train_macro_batches)
            np.random.shuffle(perm_batch)
            self._batch_list = self._batch_list[perm_batch]
            if self._data_type == 'FBL':
                self._current_macro_batch, self._current_label = read_macro_batch_FBL(
                    self._batch_list[start_point_macro_batch], self._sample_size, shuffle)
            elif self._data_type == 'FWL':
                self._current_macro_batch, self._current_label = read_macro_batch_FWL(
                    self._batch_list[start_point_macro_batch], self._sample_size, shuffle)
            else:
                self._current_macro_batch, self._current_label = read_macro_batch(
                    self._batch_list[start_point_macro_batch], self._sample_size, shuffle)
        # go to next macro batch
        if index_in_macro_batch + batch_size > self._num_trials_in_macro_batch:
            if start_point_macro_batch < self._num_train_macro_batches:   # haven't reach the end of the epoch
                start_point_macro_batch += 1
            if start_point_macro_batch >= self._num_train_macro_batches:  # reach the end of the epoch, shuffle
                self._epochs_completed += 1
                start_point_macro_batch = 0
                perm_batch = np.arange(self._num_train_macro_batches)
                np.random.shuffle(perm_batch)
                self._batch_list = self._batch_list[perm_batch]
            rest_num_examples = self._num_trials_in_macro_batch - index_in_macro_batch
            data_rest_part = self._current_macro_batch[index_in_macro_batch:self._num_trials_in_macro_batch]
            label_rest_part = self._current_label[index_in_macro_batch:self._num_trials_in_macro_batch, 0]
            # read next macro batch
            if self._data_type == 'FBL':
                self._current_macro_batch, self._current_label = read_macro_batch_FBL(
                    self._batch_list[start_point_macro_batch], self._sample_size, shuffle)
            elif self._data_type == 'FWL':
                self._current_macro_batch, self._current_label = read_macro_batch_FWL(
                    self._batch_list[start_point_macro_batch], self._sample_size, shuffle)
            else:
                self._current_macro_batch, self._current_label = read_macro_batch(
                    self._batch_list[start_point_macro_batch], self._sample_size, shuffle)
            # concatenate the trials
            index_in_macro_batch = 0
            end_point_in_macro_batch = batch_size - rest_num_examples
            self._macro_batch_index_in_epoch = start_point_macro_batch
            self._index_in_macro_batch = end_point_in_macro_batch
            new_data = self._current_macro_batch[index_in_macro_batch:end_point_in_macro_batch]
            new_label = self._current_label[index_in_macro_batch:end_point_in_macro_batch, 0]
            return np.concatenate((data_rest_part, new_data), axis=0), np.concatenate((label_rest_part, new_label),
                                                                                      axis=0)
        else:
            self._index_in_macro_batch += batch_size
            end_point_in_macro_batch = self._index_in_macro_batch
            return self._current_macro_batch[index_in_macro_batch:end_point_in_macro_batch], self._current_label[
                index_in_macro_batch:end_point_in_macro_batch, 0]




