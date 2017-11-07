from lasagne.layers import (
    InputLayer, DenseLayer,
    Conv2DLayer, Pool2DLayer,
    get_all_params, get_all_param_values,
    set_all_param_values, get_output, batch_norm)
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.nonlinearities import softmax
from lasagne.updates import adam

from theano import tensor as T
import theano

from matplotlib import pyplot as plt
from collections import OrderedDict
from argparse import ArgumentParser
from queue import PriorityQueue
from functools import reduce
import numpy as np
import pickle
import os


early_stopping_epochs = 3
hidden_size = 4096
max_epochs = 20


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
        hidden_size=hidden_size, output_classes=10):
    network = OrderedDict()

    def previous_layer():
        return network[next(reversed(network))]

    def apply_batch_norm():
        last_layer_key = next(reversed(network))
        network[last_layer_key] = batch_norm(
            network[last_layer_key]
        )

    # network['input'] = InputLayer(
    #     shape=(batch_size, feature_dimensionality),
    #     input_var=input_var
    # )
    network['input'] = InputLayer(
        shape=(batch_size, 3, 32, 32),
        input_var=input_var
    )

    network['conv1_1'] = Conv2DLayer(
        previous_layer(),
        num_filters=64,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv1_2'] = Conv2DLayer(
        previous_layer(),
        num_filters=64,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['pool1'] = Pool2DLayer(
        previous_layer(),
        2
    )

    network['conv2_1'] = Conv2DLayer(
        previous_layer(),
        num_filters=128,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv2_2'] = Conv2DLayer(
        previous_layer(),
        num_filters=128,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['pool2'] = Pool2DLayer(
        previous_layer(),
        2
    )

    network['conv3_1'] = Conv2DLayer(
        previous_layer(),
        num_filters=256,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv3_2'] = Conv2DLayer(
        previous_layer(),
        num_filters=256,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv3_3'] = Conv2DLayer(
        previous_layer(),
        num_filters=256,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['pool3'] = Pool2DLayer(
        previous_layer(),
        2
    )

    network['conv4_1'] = Conv2DLayer(
        previous_layer(),
        num_filters=512,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv4_2'] = Conv2DLayer(
        previous_layer(),
        num_filters=512,
        filter_size=(3, 3),

    apply_batch_norm()
    network['conv4_3'] = Conv2DLayer(
        previous_layer(),
        num_filters=512,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['pool4'] = Pool2DLayer(
        previous_layer(),
        2
    )

    network['conv5_1'] = Conv2DLayer(
        previous_layer(),
        num_filters=512,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv5_2'] = Conv2DLayer(
        previous_layer(),
        num_filters=512,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['conv5_3'] = Conv2DLayer(
        previous_layer(),
        num_filters=512,
        filter_size=(3, 3),
        pad='same',
    )
    apply_batch_norm()
    network['pool5'] = Pool2DLayer(
        previous_layer(),
        2
    )

    network['fc6'] = DenseLayer(
        previous_layer(),
        512
    )
    apply_batch_norm()

    network['output'] = DenseLayer(
        previous_layer(),
        output_classes,
        nonlinearity=softmax
    )

    return network


def get_k_network_initialisations(k, *args, **kwargs):
    params_list = []
    for index in range(k):
        network = build_network(*args, **kwargs)
        params = get_all_param_values(network['output'])
        params_list.append(params)

    return params_list


def get_bootstrap(*input_arrays):
    row_indices = np.arange(input_arrays[0].shape[0])
    bootstrap_indices = np.random.choice(
        row_indices, size=row_indices.size
    )

    return tuple(
        input_array[bootstrap_indices]
        for input_array in input_arrays
    )


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
        # train_data = train_data.transpose(0, 2, 3, 1)

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


def make_training_function(
        train_function, loss_function,
        accuracy_function, network,
        val_X, val_y):
    def train_network(
            train_X, train_y, initial_network_params,
            print_train_info=True, plot_figures=True):

        set_all_param_values(
            network['output'],
            initial_network_params
        )

        n_chunks = 225
        train_chunks = list(zip(
            np.split(train_X, n_chunks),
            np.split(train_y, n_chunks)
        ))
        if print_train_info:
            sizes = set(chunk[0].shape[0] for chunk in train_chunks)
            print('Minibatch sizes used: {}'.format(sizes))

        training_losses = []
        validation_losses = []
        validation_accuracies = []
        min_val_loss = None

        param_queue = PriorityQueue(maxsize=2)
        for epoch in range(1, max_epochs):
            training_loss = 0.
            for X_chunk, y_chunk in train_chunks:
                minibatch_loss = train_function(X_chunk, y_chunk) / n_chunks
                # print(minibatch_loss)
                training_loss += minibatch_loss

            validation_loss = float(loss_function(val_X, val_y))
            validation_accuracy = float(accuracy_function(val_X, val_y))

            if print_train_info:
                print('Epoch {}\n\ttrain loss: {}'.format(
                    epoch, training_loss))
                print('\tval loss: {}'.format(validation_loss))
                print('\tval accuracy: {:.2f}%'.format(validation_accuracy))

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            current_params = get_all_param_values(network['output'])
            if param_queue.full():
                param_queue.get()

            param_queue.put((-validation_loss, current_params))

            if min_val_loss is None or validation_loss <= min_val_loss:
                min_val_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == early_stopping_epochs:
                if print_train_info:
                    print('Breaking due to early stopping!')
                break

        while param_queue.qsize() > 0:
            key, best_params = param_queue.get()

        if plot_figures:
            plt.figure()
            plt.plot(*zip(*enumerate(training_losses, 1)))
            plt.plot(*zip(*enumerate(validation_losses, 1)))
            plt.show()

        return (
            best_params, training_losses,
            validation_losses, validation_accuracies
        )
    return train_network


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


def main():
    args = build_parser().parse_args()

    dataset_mean = load_mean(args.mean_path)
    X, y = load_data(
        args.dataset_directory, dataset_mean,
        mean_normalise=True, four_dim=True)

    train_X, train_y, val_X, val_y = train_val_split(X, y)

    print([mat.shape for mat in (train_X, train_y, val_X, val_y)])

    # input_var = T.matrix('input', dtype=theano.config.floatX)
    input_var = T.tensor4('input', dtype=theano.config.floatX)
    target = T.vector('target', dtype='int32')

    network = build_network(input_var=input_var)
    prediction = get_output(network['output'])
    loss = categorical_crossentropy(prediction, target).mean()
    accuracy = np.array(100., dtype=theano.config.floatX) * (
        categorical_accuracy(prediction, target).mean())

    params = get_all_params(network['output'], trainable=True)
    updates = adam(loss, params)

    print('Starting theano function compliation')
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
    pred_function = theano.function(
        [input_var],
        prediction
    )
    print('Finished theano function compliation')
    ensemble_prediction = make_ens_predictor(
        network, pred_function, val_X, val_y
    )
    train_network = make_training_function(
        train_function, loss_function,
        accuracy_function, network,
        val_X, val_y
    )

    k = 8
    initialisations = get_k_network_initialisations(k, input_var=input_var)
    bootstraps = [
        get_bootstrap(train_X, train_y)
        for _ in range(k)
    ]
    ensembles = zip(initialisations, bootstraps)

    # initial_network_params = get_all_param_values(network['output'])

    trained_parameters = []
    for initialisation, bootstrap in ensembles:
        (
            best_params, training_losses,
            validation_losses, validation_accuracies
        ) = train_network(
            *bootstrap, initialisation, True, True)
        # *bootstrap, initialisation, False, False)
        trained_parameters.append(best_params)
        max_accuracy = np.max(validation_accuracies)
        # print(np.min(validation_losses))
        ensemble_accuracy = ensemble_prediction(trained_parameters)
        print('New member at {:.2f}% validation accuracy'.format(max_accuracy))
        print(
            'Ensemble at {:.2f}% with {} members'
            ''.format(ensemble_accuracy, len(trained_parameters))
        )
        print()
        import IPython
        IPython.embed()

    ensemble_accuracy = ensemble_prediction(trained_parameters)
    print(ensemble_accuracy)


if __name__ == '__main__':
    main()
