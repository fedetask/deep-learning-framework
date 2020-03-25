"""
This module defines the elements of a computation graph.
"""

from abc import ABC, abstractmethod
import numpy as np


class ComputationNode:
    """
    This class represent the higher abstraction of a node in the computation graph.
    """

    def __init__(self, name='""'):
        self.childs = []
        self.parents = []
        self.output_shape = None
        self.name = name

    def __call__(self, parents):
        """Returns the ComputationNode, set to operate on the given parent nodes

        Args:
            parents (list or ComputationNode): can be a single or a list of ComputationNode

        Returns:
            A ComputationNode object set with the given parents

        """
        if isinstance(parents, ComputationNode):  # If called on single node
            self.parents = [parents]
        elif isinstance(parents, list):
            self.parents = parents
        else:
            raise ValueError('ComputationNode can be called only on'
                             ' a ComputationNode or a list of ComputationNode')
        for parent in self.parents:
            parent.childs.append(self)
        return self

    @abstractmethod
    def eval(self):
        """Evaluate the node on real values.

        Each derived class must provide an appropriate implementation depending on the operation
        it performs.

        Returns:
            A numpy array with the result of the computation.
        """
        pass

    def __str__(self):
        return str(type(self)) + ', name: ' + self.name + ', out_shape: ' + str(self.output_shape)


class DotProduct(ComputationNode):
    """
    This node performs the dot product of its parent nodes, in the order they are given.
    """

    def __call__(self, parents):
        """Returns the node set to perform the dot product operation on the two parent nodes

        Args:
            parents (list): List of two ComputationNode objects on which the dot product is defined

        Returns:
            A DotProduct node defined on the given parent nodes

        """
        assert isinstance(parents, list), 'DotProduct can be called only on a list'
        assert len(parents) == 2, 'Error: can only perform dot product on 2 inputs'
        self.output_shape = [parents[0].output_shape[0], parents[1].output_shape[1]]
        return super(DotProduct, self).__call__(parents)

    def eval(self):
        a = self.parents[0].eval()
        b = self.parents[1].eval()
        return np.dot(a, b)


class ValueNode(ComputationNode):
    """
    This class represent a container node that keeps numerical values. It does not perform
    operations and cannot be called on other nodes, but can be trained and other nodes can be
    called on this.
    """

    def __init__(self, shape, name=None):
        super(ValueNode, self).__init__(name=name)
        self.output_shape = shape
        self.values = None

    def set_values(self, values):
        shape_error_msg = 'Values shape ' + str(values.shape) + 'does not match with node shape '\
                                                              + str(self.output_shape)
        if self.output_shape[0] is None:
            assert self.output_shape[1:] == values.shape[1:], shape_error_msg
        else:
            assert self.output_shape == values.shape, shape_error_msg
        self.values = values

    def eval(self):
        return self.values


if __name__ == '__main__':
    x = ValueNode(shape=(None, (3, 3)), name='x')
    x.set_values(np.array([
        [[1, 2], [1, 2], []],
    ]))
    W = ValueNode(shape=(7, 2), name='W')
    W.set_values(np.ones(shape=(7, 2)))

    p = DotProduct()([x, W])
    res = p.eval()
    print(res)
