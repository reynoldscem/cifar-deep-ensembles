from util import load_mean, load_data, train_val_split, make_ens_predictor

from lasagne.layers import get_output

from theano import tensor as T
import theano

from argparse import ArgumentParser
from itertools import combinations
import numpy as np
import json
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
        '-e', '--experiment-path',
        help='Path to bagging experiment output.',
        required=True
    )

    return parser


def main():
    args = build_parser().parse_args()
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

    network_kwargs = {
        'input_var': input_var,
        'base_power': args.base_power
    }
    model = MiniVGG(**network_kwargs)
    model.pretty_print_network()

    network = model.network
    prediction = get_output(network['output'])

    print('Starting theano function compliation')
    pred_function = theano.function(
        [input_var],
        prediction
    )
    print('Finished theano function compliation')
    ensemble_prediction = make_ens_predictor(
        network, pred_function, val_X, val_y
    )

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
