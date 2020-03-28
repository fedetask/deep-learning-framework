"""
This module provides implementation of computation layers. A layer is a wrapping around a sub-graph
in the computation graph that is of particular utility. E.g. the Dense layer is a wrapper around

weights  ------ z ----- activation
                |
biases   -------+
"""

from abc import ABC, abstractmethod
from computation_graph import ComputationNode, Values, DotProduct, Sum, Softmax
from activations import get_activation
from initializers import get_initializer
import numpy as np


class Layer(ABC):

    @abstractmethod
    def __call__(self, parents):
        """Instantiate the ComputationNode objects provided by the layer.

        When a layer derived from this class is called on a ComputationNode or a list of
        ComputationNode objects it must create the appropriate ComputationNode objects that the
        Layer implementation provides, call the first node(s) on the input node(s), and return
        the last node(s) of the layer.

        Args:
            parents (Union[ComputationNode, list]): The nodes required by the derived class to
            act as layer input

        Returns:
            Must return the layer output node or a list of output nodes
        """
        pass


class Dense(Layer):
    """Implementation of the fully-connected layer. Works only for batches of vector inputs.
    """

    initializer_for_activation = {'relu': 'he', 'softmax': 'xavier', None: 'xavier'}

    def __init__(self, units, activation=None, initializer='auto'):
        self.units = units
        self.weights = None
        self.biases = None
        self.activation = get_activation(activation)
        if initializer == 'auto':
            self.weights_initializer = get_initializer(Dense.initializer_for_activation[activation])
        else:
            self.weights_initializer = get_initializer(initializer)
        self.bias_initializer = get_initializer('zero')

    def __call__(self, inputs):
        self._check_input(inputs)

        # Create the weights node.
        if isinstance(inputs, list):
            n_rows = sum(i.shape[-1] for i in inputs)
        else:
            n_rows = inputs.shape[-1]
        self.weights = Values(shape=(n_rows, self.units), trainable=True)

        # Dot product with the inputs
        dot = DotProduct()([inputs, self.weights])

        # Create the biases
        if dot.shape[0] is None:
            biases_shape = dot.shape[1:]
        else:
            biases_shape = dot.shape
        self.biases = Values(shape=biases_shape, trainable=True)

        sum_biases = Sum()([dot, self.biases])

        self._init_weights()

        if self.activation is not None:
            return self.activation()(sum_biases)
        else:
            return sum_biases

    def _init_weights(self):
        self.weights_initializer(self.weights)
        self.bias_initializer(self.biases)

    def _check_input(self, inputs):
        """Ensures that all inputs have the same first shape

        Args:
            inputs (ComputationNode or list of ComputationNode): Node(s) that act as input for
            the Dense layer

        Raises:
            AssertionError if input is not a ComputationNode nor a list of ComputationNode.
            AssertionError if input is a list of ComputationNode and not all nodes have the same
            first shape.

        """
        if isinstance(inputs, ComputationNode):  # A single ComputationNode is always valid
            return
        if isinstance(inputs, list):
            raise NotImplementedError('Dense on multiple layers is not implemented yet.')
        shape = inputs[0].shape
        for input_node in inputs:
            assert isinstance(input_node, ComputationNode)
            assert input_node.shape[0] == shape[0]


if __name__ == '__main__':
    x = Values(shape=(None, 5))
    l = Dense(units=16)(x)
    print('layer shape ' + str(l.shape))