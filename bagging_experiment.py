from lasagne.layers import (
    get_all_params, get_all_param_values,
    set_all_param_values, get_output)
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.updates import adam

from theano import tensor as T
import theano

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from itertools import combinations
from queue import PriorityQueue
from functools import reduce
import numpy as np
import subprocess
import pickle
import json
import time
import sys
import os

from mini_vgg import MiniVGG


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

    parser.add_argument(
        '--early-stopping-epochs',
        help=(
            'Number of epochs without improvement for which to train.'
            ' Negative values will prevent using early stopping.'
        ),
        type=int,
        default=3
    )

    parser.add_argument(
        '--max-epochs',
        help='Max number of epochs to train for',
        type=int,
        default=50
    )

    parser.add_argument(
        '-k', '--num-individuals',
        help='Number of individuals to train for ensemble',
        type=int,
        default=128
    )

    parser.add_argument(
        '--base-power',
        help=(
            'Denotes the starting number of filters.'
            'First block is 2**base_power. Doubles each block.'
        ),
        type=int,
        default=3
    )

    parser.add_argument(
        '--seed',
        help='Random seed for numpy to use.',
        type=int,
        default=72826
    )

    return parser


def get_k_network_initialisations(k, *args, **kwargs):
    params_list = []
    for index in range(k):
        model = MiniVGG(*args, **kwargs)
        network = model.network
        params = get_all_param_values(network['output'])
        params_list.append(params)

    return params_list


def get_bootstrap(*input_arrays):
    row_indices = np.arange(input_arrays[0].shape[0])
    bootstrap_indices = np.random.choice(
        row_indices, size=row_indices.size
    )

    return (
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
        val_X, val_y, max_epochs, early_stopping_epochs):
    def train_network(
            train_X, train_y, initial_network_params,
            print_train_info=True, plot_figures=True):

        set_all_param_values(
            network['output'],
            initial_network_params
        )

        n_chunks = 100
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
        for epoch in range(1, max_epochs + 1):
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
                sys.stdout.flush()

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


# From https://stackoverflow.com/questions/3431825/ ...
# generating-an-md5-checksum-of-a-file
def md5(filename):
    from hashlib import md5
    hash_md5 = md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    args = build_parser().parse_args()

    assert args.num_individuals >= 1, (
        'Must have at least one member in ensemble'
    )
    assert args.max_epochs >= 1, (
        'Must have at least 1 epoch.'
    )

    assert args.base_power >= 0, (
        'Cannot have fractional filters!'
    )

    np.random.seed(args.seed)
    import lasagne
    lasagne.random.set_rng(np.random.RandomState(args.seed))
    experiment_timestamp = str(time.time()).replace('.', '-')
    experiment_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'experiments',
        experiment_timestamp
    )
    if os.path.exists(experiment_path):
        print('Experiment directory exists!')
        sys.exit(1)
    else:
        os.makedirs(experiment_path)

    # Save the commit hash used for these experiments.
    commit_hash = str(
        subprocess.check_output(['git', 'rev-parse', 'HEAD']),
        'utf-8'
    )
    commit_file_path = os.path.join(experiment_path, 'exp_commit.txt')
    with open(commit_file_path, 'w') as fd:
        fd.write(commit_hash)

    args_file_path = os.path.join(experiment_path, 'provided_args.txt')
    with open(args_file_path, 'w') as fd:
        json.dump(vars(args), fd, indent=4)

    # Initial dataset setup
    dataset_mean = load_mean(args.mean_path)
    X, y = load_data(
        args.dataset_directory, dataset_mean,
        mean_normalise=True, four_dim=True)

    train_X, train_y, val_X, val_y = train_val_split(X, y)

    print(
        'Train X shape: {}\ttrain y shape: {}'
        'Test X shape: {}\tTest y shape: {}'
        ''.format(*(mat.shape for mat in (train_X, train_y, val_X, val_y)))
    )

    # Network setup
    input_var = T.tensor4('input', dtype=theano.config.floatX)
    target = T.vector('target', dtype='int32')

    network_kwargs = {
        'input_var': input_var,
        'base_power': args.base_power
    }
    model = MiniVGG(**network_kwargs)
    model.pretty_print_network()

    network = model.network
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
        val_X, val_y,
        args.max_epochs,
        args.early_stopping_epochs
    )

    # Setup bootstraps
    initialisations = get_k_network_initialisations(
        args.num_individuals,
        input_var=input_var, base_power=args.base_power
    )
    bootstraps = [
        get_bootstrap(train_X, train_y)
        for _ in range(args.num_individuals)
    ]
    ensembles = zip(initialisations, bootstraps)

    # Train models
    trained_parameters = []
    for index, (initialisation, bootstrap) in enumerate(ensembles):
        (
            best_params, training_losses,
            validation_losses, validation_accuracies
        ) = train_network(
            *bootstrap, initialisation, True, False)
        trained_parameters.append(best_params)

        max_accuracy = validation_accuracies[np.argmin(validation_losses)]
        ensemble_accuracy = ensemble_prediction(trained_parameters)

        print('New member at {:.2f}% validation accuracy'.format(max_accuracy))
        print(
            'Ensemble at {:.2f}% with {} members'
            ''.format(ensemble_accuracy, len(trained_parameters))
        )
        print()
        sys.stdout.flush()

        member_path = os.path.join(experiment_path, 'model_{}'.format(index))
        os.makedirs(member_path)
        stats = {
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'validation_accuracies': validation_accuracies
        }
        with open(os.path.join(member_path, 'train_stats.json'), 'w') as fd:
            json.dump(stats, fd, indent=4)
        model_save_path = os.path.join(member_path, 'model.npz')
        np.savez(
            model_save_path,
            *get_all_param_values(model.final_layer)
        )
        model_hash = md5(model_save_path)
        model_hash_path = os.path.join(member_path, 'model_hash.txt')
        with open(model_hash_path, 'w') as fd:
            fd.write(model_hash + '\n')

    ensemble_accuracies = {}
    for num_models in range(1, args.num_individuals + 1):
        parameter_combinations = combinations(
            trained_parameters, num_models
        )
        validation_accuracies = [
            ensemble_prediction(parameter_combination)
            for parameter_combination in parameter_combinations
        ]
        ensemble_accuracies[num_models] = {
            'mean': np.mean(validation_accuracies),
            'std': np.std(validation_accuracies),
            'raw': validation_accuracies
        }
    results_path = os.path.join(experiment_path, 'results.json')
    with open(results_path, 'w') as fd:
        json.dump(ensemble_accuracies, fd, indent=4)


if __name__ == '__main__':
    main()
