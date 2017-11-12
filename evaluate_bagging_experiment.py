from util import load_mean, load_data, train_val_split, make_ens_predictor

from lasagne.layers import get_output

from theano import tensor as T
import theano

from argparse import ArgumentParser
import numpy as np
import json
import os

from random import sample
import sys

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


def load_experiment_params(experiment_path):
    # Commit 0aff288422aba2ed485333badf821f6a022c4aef writes provided_args
    #  with the extension .txt, so account for both.
    matching_filenames = [
        filename
        for filename in os.listdir(experiment_path)
        if filename.startswith('provided_args')
    ]
    if len(matching_filenames) == 0:
        raise FileNotFoundError('Couldn\'t find argument file!')
    else:
        filename = os.path.join(experiment_path, matching_filenames[0])

    with open(filename) as fd:
        experiment_params = dict(json.load(fd))

    return experiment_params


def load_weights(experiment_path, num_individuals):
    def load_model_file(path):
        with np.load(path) as fd:
            params = [
                fd['arr_{}'.format(index)]
                for index in range(len(fd.files))
            ]
        return params
    paths = [
        os.path.join(experiment_path, 'model_{}'.format(index), 'model.npz')
        for index in range(num_individuals)
    ]
    parameters = [load_model_file(path) for path in paths]
    return parameters


def main():
    args = build_parser().parse_args()

    assert os.path.exists(args.experiment_path), (
        'Experiment path must exist!'
    )

    experiment_params = load_experiment_params(args.experiment_path)

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
        'base_power': experiment_params['base_power']
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

    trained_parameters = load_weights(
        args.experiment_path, experiment_params['num_individuals']
    )
    num_samples = 5
    ensemble_accuracies = {}
    for num_models in range(1, experiment_params['num_individuals'] + 1):
        parameter_combinations = [
            sample(trained_parameters, num_models)
            for _ in range(num_samples)
        ]
        validation_accuracies = [
            ensemble_prediction(parameter_combination)
            for parameter_combination in parameter_combinations
        ]
        ensemble_accuracies[num_models] = {
            'mean': np.mean(validation_accuracies),
            'std': np.std(validation_accuracies),
            'raw': validation_accuracies
        }
        print(ensemble_accuracies)
        sys.stdout.flush()
    import time
    experiment_timestamp = str(time.time()).replace('.', '-')
    results_path = os.path.join(
        args.experiment_path, 'results_{}.json'.format(experiment_timestamp))
    with open(results_path, 'w') as fd:
        json.dump(ensemble_accuracies, fd, indent=4)


if __name__ == '__main__':
    main()
