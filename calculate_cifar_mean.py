'''Script to compute mean vector for training set
 of the cifar-10 dataset python version.
'''
from argparse import ArgumentParser
import numpy as np
import pickle
import os


min_batch_index, max_batch_index = 1, 6
batch_format_string = 'data_batch_{}'
normalisation_constant = 255.
outfile_name = 'cifar_mean.npy'


def build_parser():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        '-d', '--dataset-directory',
        help='Directory containing cifar-10.',
        required=True
    )

    return parser


def make_accumulator_func(dataset_directory):
    def accumulator(index):
        path = os.path.join(
            dataset_directory,
            batch_format_string.format(index)
        )

        with open(path, 'rb') as fd:
            data_dict = pickle.load(fd, encoding='bytes')

        data = data_dict[b'data']
        normalised = data.astype(np.float32) / normalisation_constant
        mean = np.mean(normalised, axis=0)

        return mean

    return accumulator


def main():
    args = build_parser().parse_args()

    assert os.path.isdir(args.dataset_directory), (
        'Data directory must exist.'
    )

    assert not os.path.isfile(outfile_name), (
        '{} already exists!'.format(outfile_name)
    )

    accumulator = make_accumulator_func(args.dataset_directory)

    dataset_mean = np.mean(
        [
            accumulator(index)
            for index in range(min_batch_index, max_batch_index)
        ],
        axis=0
    )

    print('Mean vector norm:\t{:.4f}'.format(np.linalg.norm(dataset_mean)))
    print('Mean vector length:\t{}'.format(dataset_mean.shape))

    np.save(outfile_name, dataset_mean)


if __name__ == '__main__':
    main()
