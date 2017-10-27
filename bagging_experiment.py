from lasagne.layers import (
    InputLayer, DenseLayer, get_all_params, get_all_param_values, get_output)
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import adam

from theano import tensor as T
import theano

from collections import OrderedDict
from argparse import ArgumentParser
import numpy as np
import pickle
import os


def build_parser():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        '-d', '--dataset-directory',
        help='Directory containing cifar-10.',
        required=True
    )

    return parser


def build_network(
        input_var=None,
        batch_size=None, feature_dimensionality=3072,
        hidden_size=4000, output_classes=10):
    network = OrderedDict()

    network['input'] = InputLayer(
        shape=(batch_size, feature_dimensionality),
        input_var=input_var
    )

    network['fc1'] = DenseLayer(
        network['input'], hidden_size
    )

    network['output'] = DenseLayer(
        network['fc1'],
        output_classes,
        nonlinearity=softmax
    )

    return network


def get_bootstrap(*input_arrays):
    row_indices = np.arange(input_arrays[0].shape[0])
    bootstrap_indices = np.random.choice(
        row_indices, size=row_indices.size
    )

    return tuple(
        input_array[bootstrap_indices]
        for input_array in input_arrays
    )


def load_data(dataset_directory, dataset_mean):
    def load_index(index):
        path = os.path.join(
            dataset_directory,
            'data_batch_{}'.format(index)
        )

        with open(path, 'rb') as fd:
            data_dict = pickle.load(fd, encoding='bytes')

        return tuple(data_dict[key] for key in (b'data', b'labels'))

    data_chunks = [load_index(index) for index in range(1, 6)]
    train_data, train_labels = (
        np.concatenate([chunk[chunk_index] for chunk in data_chunks])
        for chunk_index in range(2)
    )

    assert train_data.shape[0] == train_labels.shape[0]
    train_data = train_data / 255.

    assert train_data.shape[1] == dataset_mean.shape[0]
    train_data = train_data - dataset_mean

    return train_data, train_labels


def train_val_split(X, y, val_proportion=0.1):
    split_index = np.floor(X.shape[0] * (1 - val_proportion)).astype(np.int)

    train_X, val_X = X[:split_index], X[split_index:]
    train_y, val_y = y[:split_index], y[split_index:]

    return train_X, train_y, val_X, val_y


def main():
    args = build_parser().parse_args()
    X, y = load_data(args.dataset_directory, np.zeros(3072))

    train_X, train_y, val_X, val_y = train_val_split(X, y)

    print([mat.shape for mat in (train_X, train_y, val_X, val_y)])

    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
