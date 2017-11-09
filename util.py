from lasagne.layers import set_all_param_values
from functools import reduce
import numpy as np
import pickle
import theano
import os


def load_data(
        dataset_directory, dataset_mean,
        mean_normalise=True, four_dim=True):
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

    if mean_normalise:
        train_data = train_data - dataset_mean

    if four_dim:
        train_data = train_data.reshape((-1, 3, 32, 32))

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


def make_ens_predictor(network, pred_function, X, y):
    def ensemble_prediction(parameter_list):
        probability_list = []
        for parameter_set in parameter_list:
            set_all_param_values(
                network['output'],
                parameter_set
            )
            predicted_probs = pred_function(X)
            probability_list.append(predicted_probs)
        combined_probabilities = reduce(np.add, probability_list)
        soft_vote = np.argmax(combined_probabilities, axis=1)
        accuracy = 100. * np.mean(soft_vote == y)

        return accuracy
    return ensemble_prediction
