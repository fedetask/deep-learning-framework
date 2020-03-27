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
    """
    Implementation of the fully-connected layer. Works only for batches of vector inputs.
    """

    def __init__(self, units, activation=None):
        self.units = units
        self.weights = None
        self.biases = None
        self.activation = get_activation(activation)

    def __call__(self, inputs):
        self._check_input(inputs)

        # Initialize the weight matrix.
        if isinstance(inputs, list):
            n_rows = sum(i.shape[-1] for i in inputs)
        else:
            n_rows = inputs.shape[-1]
        self.weights = Values(shape=(n_rows, self.units))
        print('setting weights shape as ' + str(self.weights.shape))
        self.weights.set_values(np.random.normal(0, 1, self.weights.shape))  # TODO temporary here
        dot = DotProduct()([inputs, self.weights])
        print('setting dot shape as ' + str(dot.shape))
        if dot.shape[0] is None:
            self.biases = Values(shape=dot.shape[1:])
        else:
            self.biases = Values(shape=dot.shape)
        print('setting biases shape as ' + str(self.biases.shape))
        sum_biases = Sum()([dot, self.biases])
        print('setting sum shape as ' + str(sum_biases.shape))
        if self.activation is not None:
            return self.activation()(sum_biases)
        else:
            return sum_biases

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
        shape = inputs[0].shape
        for input_node in inputs:
            assert isinstance(input_node, ComputationNode)
            assert input_node.shape[0] == shape[0]


if __name__ == '__main__':
    x = Values(shape=(None, 1, 5))
    l = Dense(units=16)(x)
    print('layer shape ' + str(l.shape))