from lasagne.layers import (
    InputLayer, DenseLayer,
    Conv2DLayer, Pool2DLayer,
    batch_norm, get_output_shape
)
from lasagne.nonlinearities import softmax
from collections import OrderedDict


class MiniVGG():
    def __init__(
            self, input_var=None, batch_size=None,
            output_classes=10, base_power=3):
        self.build_network(
            input_var, batch_size, output_classes, base_power
        )

    def pretty_print_network(self):
        info_list = []
        working_layer = self.final_layer

        while True:
            if isinstance(
                    working_layer,
                    (InputLayer, DenseLayer, Conv2DLayer)):
                if isinstance(working_layer, InputLayer):
                    info = (
                        'Layer name: {}\n'
                        '\tinput shape: {}\n'
                        '\toutput shape {}'
                        ''.format(
                            working_layer.name,
                            working_layer.shape,
                            get_output_shape(working_layer))
                    )
                elif isinstance(working_layer, Conv2DLayer):
                    name = working_layer.name
                    input_shape = working_layer.input_shape
                    output_shape = get_output_shape(working_layer)
                    num_filters = working_layer.num_filters
                    filter_size = working_layer.filter_size
                    pad = working_layer.pad
                    stride = working_layer.stride
                    info = (
                        'Layer name: {}\n'
                        '\tinput shape: {}\n'
                        '\toutput shape: {}\n'
                        '\tnum filters: {}\n'
                        '\tkernel shape: {}\n'
                        '\tpadding: {}\n'
                        '\tstride: {}'
                        ''.format(
                            name, input_shape, output_shape, num_filters,
                            filter_size, pad, stride
                        )
                    )
                elif isinstance(working_layer, DenseLayer):
                    name = working_layer.name
                    input_shape = working_layer.input_shape
                    output_shape = get_output_shape(working_layer)
                    num_units = working_layer.num_units
                    info = (
                        'Layer name: {}\n'
                        '\tinput shape: {}\n'
                        '\toutput shape: {}\n'
                        '\tnum units: {}'
                        ''.format(
                            name, input_shape, output_shape, num_units,
                        )
                    )
                info_list.append(info)

            if not hasattr(working_layer, 'input_layer'):
                break
            working_layer = working_layer.input_layer

        for item in reversed(info_list):
            print(item)
            print()

        depth = len(info_list) - 1
        print('Total network depth {} layers'.format(depth))
        print('Depth is total of convolutional and fully connected layers.')

    def build_network(
            self, input_var=None, batch_size=None,
            output_classes=10, base_power=3):
        network = OrderedDict()

        def previous_layer():
            return network[next(reversed(network))]

        def apply_batch_norm():
            last_layer_key = next(reversed(network))
            network[last_layer_key] = batch_norm(
                network[last_layer_key]
            )

        key = 'input'
        network[key] = InputLayer(
            shape=(batch_size, 3, 32, 32),
            input_var=input_var,
            name=key
        )

        key = 'conv1_1'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** base_power,
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv1_2'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** base_power,
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        network['pool1'] = Pool2DLayer(
            previous_layer(),
            2,
            name=key
        )

        key = 'conv2_1'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 1),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv2_2'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 1),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'pool2'
        network[key] = Pool2DLayer(
            previous_layer(),
            2,
            name=key
        )

        key = 'conv3_1'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 2),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv3_2'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 2),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv3_3'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 2),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'pool3'
        network[key] = Pool2DLayer(
            previous_layer(),
            2,
            name=key
        )

        key = 'conv4_1'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv4_2'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv4_3'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'pool4'
        network[key] = Pool2DLayer(
            previous_layer(),
            2,
            name=key
        )

        key = 'conv5_1'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv5_2'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'conv5_3'
        network[key] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
            name=key
        )
        apply_batch_norm()
        key = 'pool5'
        network[key] = Pool2DLayer(
            previous_layer(),
            2,
            name=key
        )

        key = 'fc6'
        network[key] = DenseLayer(
            previous_layer(),
            2 ** (base_power + 3),
            name=key
        )
        apply_batch_norm()

        key = 'output'
        network[key] = DenseLayer(
            previous_layer(),
            output_classes,
            nonlinearity=softmax,
            name=key
        )

        self.final_layer = previous_layer()
        self.network = network

        return network
