from lasagne.layers import (
    InputLayer, DenseLayer,
    Conv2DLayer, Pool2DLayer,
    batch_norm
)
from lasagne.nonlinearities import softmax
from collections import OrderedDict


class MiniVGG():
    def __init__(
            self, input_var=None, batch_size=None,
            output_classes=10, base_power=3):
        self.network = self.build_network(
            input_var, batch_size, output_classes, base_power
        )

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

        network['input'] = InputLayer(
            shape=(batch_size, 3, 32, 32),
            input_var=input_var
        )

        network['conv1_1'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** base_power,
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv1_2'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** base_power,
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
            num_filters=2 ** (base_power + 1),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv2_2'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 1),
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
            num_filters=2 ** (base_power + 2),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv3_2'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 2),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv3_3'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 2),
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
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv4_2'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv4_3'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
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
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv5_2'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
            filter_size=(3, 3),
            pad='same',
        )
        apply_batch_norm()
        network['conv5_3'] = Conv2DLayer(
            previous_layer(),
            num_filters=2 ** (base_power + 3),
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
            2 ** (base_power + 3)
        )
        apply_batch_norm()

        network['output'] = DenseLayer(
            previous_layer(),
            output_classes,
            nonlinearity=softmax
        )

        return network
