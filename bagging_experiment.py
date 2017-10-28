from lasagne.layers import (
    InputLayer, DenseLayer, get_all_params, get_all_param_values, get_output)
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
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

    parser.add_argument(
        '-m', '--mean-path',
        help='Path to npy file containing dataset mean.',
        default='cifar_mean.npy'
    )

    return parser


def build_network(
        input_var=None,
        batch_size=None, feature_dimensionality=3072,
        hidden_size=1000, output_classes=10):
    network = OrderedDict()

    network['input'] = InputLayer(
        shape=(batch_size, feature_dimensionality),
        input_var=input_var
    )

    network['fc1'] = DenseLayer(
        network['input'], hidden_size
    )
    network['fc2'] = DenseLayer(
        network['fc1'], hidden_size
    )

    network['output'] = DenseLayer(
        network['fc2'],
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

    train_data = train_data.astype(theano.config.floatX)
    train_labels = train_labels.astype(np.int32)
    return train_data, train_labels


def load_mean(mean_path):
    abs_mean_path = os.path.abspath(mean_path)
    assert os.path.isfile(abs_mean_path), (
        '{} is not a file!'.format(abs_mean_path)
    )

    return np.load(mean_path)


def train_val_split(X, y, val_proportion=0.1):
    split_index = np.floor(X.shape[0] * (1 - val_proportion)).astype(np.int)

    train_X, val_X = X[:split_index], X[split_index:]
    train_y, val_y = y[:split_index], y[split_index:]

    return train_X, train_y, val_X, val_y


def main():
    args = build_parser().parse_args()

    dataset_mean = load_mean(args.mean_path)
    X, y = load_data(args.dataset_directory, dataset_mean)

    train_X, train_y, val_X, val_y = train_val_split(X, y)

    print([mat.shape for mat in (train_X, train_y, val_X, val_y)])

    # import IPython
    # IPython.embed()

    input_var = T.matrix('input', dtype=theano.config.floatX)
    target = T.vector('target', dtype='int32')

    network = build_network(input_var=input_var)
    prediction = get_output(network['output'])
    loss = categorical_crossentropy(prediction, target).mean()
    accuracy = categorical_accuracy(prediction, target).mean()

    params = get_all_params(network['output'], trainable=True)
    updates = adam(loss, params)

    train_function = theano.function(
        [input_var, target],
        loss,
        updates=updates
    )
    loss_function = theano.function(
        [input_var, target],
        loss
    )
    accuracy_function = theano.function(
        [input_var, target],
        accuracy
    )

    n_chunks = 50
    train_chunks = list(zip(
        np.split(train_X, n_chunks),
        np.split(train_y, n_chunks)
    ))
    epoch_losses = []
    validation_losses = []
    for epoch in range(1, 250):
        epoch_loss = 0.
        for X_chunk, y_chunk in train_chunks:
            epoch_loss += train_function(X_chunk, y_chunk) / n_chunks

        epoch_losses.append(epoch_loss)
        print('Epoch {}\n\ttrain loss: {}'.format(epoch, epoch_loss))

        validation_loss = float(loss_function(val_X, val_y))
        validation_accuracy = float(accuracy_function(val_X, val_y) * 100.)
        print('\tval loss: {}'.format(validation_loss))
        print('\tval accuracy: {:.2f}%'.format(validation_accuracy))

        validation_losses.append(validation_loss)
    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
