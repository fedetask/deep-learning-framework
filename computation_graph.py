"""
This module defines the elements of a computation graph.
"""

from abc import ABC, abstractmethod
import numpy as np


class ComputationNode(ABC):
    """
    This class represent the higher abstraction of a node in the computation graph. A node can
    use other nodes (parents) as input, and can be used as input by other nodes (children).

    Attributes:
        children (list): List of ComputationNode objects that use this node as input. This is
            generally set by __call__() of those nodes when they are called on this.
        parents (list): List of ComputationNode objects that this node uses as input. This is
            generally set by __call__() of this node when called on parent nodes.
        shape (tuple): Provides the shape of this node to other nodes. The first element of shape
            can be None. In that case the first dimension will be determined and check at runtime.
    """

    def __init__(self, name='""'):
        self.children = []
        self.parents = []
        self.shape = None
        self.name = name

    def __call__(self, parents):
        """Returns the ComputationNode, set to operate on the given parent nodes.

        When a node is called on other nodes, those nodes become parents of the node,
        and the node will appear in their children array. The shape of the node is computed and
        set in the shape variable

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
            parent.children.append(self)

        self.shape = self._get_shape(self.parents)
        return self

    def eval(self):
        """Perform the computation that the node implements.

        Returns:
            A numpy array with the result of the computation.
        """
        parent_values = [parent.eval() for parent in self.parents]
        self._check_eval_input(parent_values)
        return self._eval_node(parent_values)

    @abstractmethod
    def _eval_node(self, parents_values):
        """Evaluate the node on the given parent values.

        Args:
            parents_values (list): List of numpy.ndarray with the result of parents evaluation.

        Returns:
            A numpy.ndarray with the value of this node, evaluated on the given parent values.

        """
        pass

    @abstractmethod
    def _check_eval_input(self, parent_values):
        """Runtime check of the parent node values. Check that shape dimensions that were not
        known at graph definitions are correct at graph execution.

        Raises:
            AssertionError if the parent(s) eval() result is not compatible with the node

        """
        pass

    @abstractmethod
    def _get_shape(self, parents):
        """Check that parents output shape is compatible with this node and compute the resulting
        output shape of this node. This check is computed at graph definition, when first
        dimensions can be None. Therefore, the resulting shape can have a first None dimension.

        Returns:
            The output shape of this node applied to the given parents.

        Raises:
            AssertionError if parent shapes are not compatible with the node implementation.

        """
        pass

    def __str__(self):
        return str(type(self)) + ', name: ' + self.name + ', out_shape: ' + str(self.shape)


class Values(ComputationNode):
    """
    This class represent a container node that keeps numerical values. It does not perform
    operations and cannot be called on other nodes, but can be trained and other nodes can be
    called on this.

    Attributes:
        values (numpy.ndarray) Numpy array containing the node values. Must be set by calling
            self.set_values(values) to ensure shape consistency.
    """

    def __init__(self, shape, name=None):
        """Set the shape of the node and calls the ComputationNode __init__()

        Args:
            shape (Union[int, tuple]): Shape of the values. First dimension can be None.
            name (str): Optional name for the node.
        """
        super(Values, self).__init__(name=name)
        self.shape = shape if isinstance(shape, tuple) else (shape, )
        self.values = None

    def set_values(self, values):
        """Set the given values in the node

        Args:
            values (numpy.ndarray): Numpy array consistent with self.shape

        """
        shape_error_msg = 'Values shape ' + str(values.shape) + 'does not match with node shape '\
                                                              + str(self.shape)
        if self.shape[0] is None:
            assert self.shape[1:] == values.shape[1:], shape_error_msg
        else:
            assert self.shape == values.shape, shape_error_msg
        self.values = values

    def __call__(self, parents):
        raise NotImplementedError('Values node is not callable.')

    def _get_shape(self, parents):
        return self.shape

    def _eval_node(self, parents_values):
        return self.values

    def _check_eval_input(self, parent_values):
        pass  # Values cannot have parents


class DotProduct(ComputationNode):
    """
    This node performs the dot product of two parent nodes, in the order they are given. The dot
    product is performed by broadcasting or element wise depending on the parent shapes.
    """

    def _eval_node(self, parents_values):
        return np.dot(parents_values[0], parents_values[1])

    def _get_shape(self, parents):
        assert len(parents) == 2
        a_shape = parents[0].shape
        b_shape = parents[1].shape
        if a_shape[0] is None and b_shape[0] is None:
            assert a_shape[2] == b_shape[1]
            return None, a_shape[1], b_shape[2]
        elif a_shape[0] is None and b_shape[0] is not None:
            assert a_shape[2] == b_shape[0]
            return None, a_shape[1], b_shape[1]
        elif a_shape[0] is not None and b_shape[0] is not None:
            assert a_shape[1] == b_shape[0]
            return a_shape[0], b_shape[1]
        else:
            raise NotImplementedError('Not implemented yet')

    def _check_eval_input(self, parent_values):
        a_shape = self.parents[0].shape
        b_shape = self.parents[1].shape
        # Check runtime shapes
        if a_shape[0] is None and b_shape[0] is None:
            assert parent_values[0].shape[0] == parent_values[1].shape[0]


class Sum(ComputationNode):
    """
    Performs the sum operation between two parent nodes. The sum is performed element wise or by
    broadcasting depending on the parent node shapes.
    """

    def _eval_node(self, parents_values):
        return parents_values[0] + parents_values[1]

    def _check_eval_input(self, parent_values):
        a_shape = self.parents[0].shape
        b_shape = self.parents[1].shape
        if a_shape[0] is None and b_shape[0] is None:
            assert parent_values[0].shape[0] == parent_values[1].shape[0]

    def _get_shape(self, parents):
        assert len(parents) == 2
        a_shape = parents[0].shape
        b_shape = parents[1].shape
        if a_shape[0] is None:
            if b_shape[0] is None:
                assert a_shape[1:] == b_shape[1:]
                return a_shape
            else:
                assert a_shape[1:] == b_shape
                return a_shape
        else:
            if b_shape[0] is None:
                assert b_shape[1:] == a_shape
                return b_shape
            else:
                assert a_shape == b_shape
                return a_shape


class Relu(ComputationNode):
    """
    This class implements the relu activation function.
    """

    def _eval_node(self, parents_values):
        return np.maximum(parents_values[0], 0)

    def _check_eval_input(self, parent_values):
        pass  # No check to be done for relu

    def _get_shape(self, parents):
        assert len(parents) == 0, 'Relu can be only called on one node'
        return parents[0].shape


class Softmax(ComputationNode):
    """
    Performs the Softmax function on the given node.
    """

    def _eval_node(self, parents_values):
        exp = np.exp(parents_values[0])
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _check_eval_input(self, parent_values):
        pass  # No check to be done here

    def _get_shape(self, parents):
        assert len(parents) == 1, 'Softmax can be called only on one node'
        return parents[0].shape


class Flatten(ComputationNode):
    """
    Flatten the parent node into a row vector. If the first dimension is None, the flattening is
    performed on the second dimension.
    """

    def _eval_node(self, parents_values):
        p = parents_values[0]
        if self.parents[0].shape[0] is None:
            return np.reshape(p, (p.shape[0], np.prod(p.shape[1:])))
        else:
            return np.reshape(p, (np.prod(p.shape),)),

    def _check_eval_input(self, parent_values):
        pass

    def _get_shape(self, parents):
        assert len(parents) == 1, 'Flatten can be called only on one node'
        if parents[0].shape[0] is None:
            return None, np.prod(parents[0].shape[1:])
        else:
            return np.prod(parents[0].shape),


if __name__ == '__main__':
    X = Values(shape=(None, 1, 10))
    X.set_values(np.ones((15, 1, 10)))

    W = Values(shape=(10, 3))  # E.g. 3 units layer
    W.set_values(np.zeros(W.shape))

    dot = DotProduct()([X, W])
    res_dot = dot.eval()

    print('Multiplying ' + str(X.shape) + ' with ' + str(W.shape))
    print('Actual values ' + str(X.values.shape) + ' and ' + str(W.values.shape))
    print('Result graph' + str(dot.shape) + ',  result actual ' + str(res_dot.shape))
    print('--------------------------------')

    b = Values(shape=(1, 3))
    b.set_values(np.ones((1, 3)))

    sum = Sum()([dot, b])
    res_sum = sum.eval()
    print('Summing ' + str(dot.shape) + ' with ' + str(b.shape))
    print('Actual values ' + str(res_dot.shape) + ' and ' + str(b.values.shape))
    print('Result graph ' + str(sum.shape) + ',  result actual ' + str(res_sum.shape))

    softmax = Softmax()(sum)
    res_soft = softmax.eval()
    print('Softmax on ' + str(sum.shape) + ', actual ' + str(res_sum.shape))
    print('Result graph ' + str(softmax.shape) + ',  result actual ' + str(res_soft.shape))
