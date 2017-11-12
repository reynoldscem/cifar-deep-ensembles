from lasagne.layers import (
    get_all_params, get_all_param_values,
    set_all_param_values, get_output)
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.updates import adam

from util import load_mean, load_data, train_val_split, make_ens_predictor, md5

from theano import tensor as T
import theano

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from itertools import combinations
from queue import PriorityQueue
import numpy as np
import subprocess
import json
import time
import sys
import os

from mini_vgg import MiniVGG


from lasagne.layers import ElemwiseSumLayer, ExpressionLayer


class NCEnsemble():

    @staticmethod
    def geometric_mean(
            incoming,
            eps=np.array(1e-8, dtype=theano.config.floatX)):
        exp_out = ExpressionLayer(
            ElemwiseSumLayer(
                [
                    ExpressionLayer(member, lambda x: T.log(x + eps))
                    for member in incoming
                ],
                coeffs=1./len(incoming)
            ),
            T.exp
        )
        Z = T.sum(get_output(exp_out), axis=1)[..., np.newaxis]
        return ExpressionLayer(exp_out, lambda x: x / Z)

    def get_loss(self, target, lbd):
        members = [
            self.ensemble['individual_{}'.format(index)]
            for index in range(self.num_individuals)
        ]
        error_term = T.mean(
            [
                categorical_crossentropy(
                    get_output(member.network['output']),
                    target
                )
                for member in members
            ],
            axis=0
        )
        diversity_term = T.mean(
            [
                categorical_crossentropy(
                    get_output(member.network['output']),
                    get_output(self.ensemble['p_bar'])
                )
                for member in members
            ],
            axis=0
        )
        loss = error_term + lbd * diversity_term

        return loss

    def __init__(self, num_individuals=2, input_var=None, **kwargs):
        self.num_individuals = num_individuals
        self.input_var = input_var
        self.ensemble = {}
        self.network = self.ensemble

        for index in range(self.num_individuals):
            new_individual = MiniVGG(input_var=input_var, **kwargs)
            self.ensemble['individual_{}'.format(index)] = new_individual

        self.ensemble['p_bar'] = self.geometric_mean(
            [
                self.ensemble['individual_{}'.format(index)].final_layer
                for index in range(self.num_individuals)
            ]
        )
        self.ensemble['output'] = self.ensemble['p_bar']
        self.final_layer = self.ensemble['output']


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
        default=32
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


def make_training_function(
        train_function, loss_function,
        accuracy_function, network,
        val_X, val_y, max_epochs, early_stopping_epochs):
    def train_network(
            train_X, train_y,
            print_train_info=True, plot_figures=True):

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

    import lasagne
    np.random.seed(args.seed)
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
        fd.write('\n'.join((__file__, commit_hash)))

    args_file_path = os.path.join(experiment_path, 'provided_args.json')
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

    for lbd_val in np.linspace(0., 1., 11):
        path_for_lambda = os.path.join(
            experiment_path, '{:.2f}'.format(lbd_val))
        os.makedirs(path_for_lambda)
        for num_individuals in range(1, args.num_individuals + 1):
            network_kwargs = {
                'input_var': input_var,
                'base_power': args.base_power,
                'num_individuals': num_individuals
            }
            model = NCEnsemble(**network_kwargs)

            network = model.network
            prediction = get_output(network['output'])
            loss = model.get_loss(
                target,
                np.array(lbd_val, dtype=theano.config.floatX)
            ).mean()
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
            print('Finished theano function compliation')

            train_network = make_training_function(
                train_function, loss_function,
                accuracy_function, network,
                val_X, val_y,
                args.max_epochs,
                args.early_stopping_epochs
            )

            (
                best_params, training_losses,
                validation_losses, validation_accuracies
            ) = train_network(
                train_X, train_y, True, False)

            ensemble_accuracy = validation_accuracies[
                np.argmin(validation_losses)]

            print(
                'Ensemble at {:.2f}% with {} members'
                ''.format(ensemble_accuracy, num_individuals)
            )
            print()
            sys.stdout.flush()

            member_path = os.path.join(
                path_for_lambda, 'ensemble_{}'.format(num_individuals))
            os.makedirs(member_path)
            stats = {
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'validation_accuracies': validation_accuracies
            }
            stats_path = os.path.join(member_path, 'train_stats.json')
            with open(stats_path, 'w') as fd:
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


if __name__ == '__main__':
    main()
