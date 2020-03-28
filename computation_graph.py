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
        name (str): Optional name of the node
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
            parents (Union[list, ComputationNode]): Can be a single or a list of ComputationNode

        Returns:
            The created ComputationNode in the graph

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

        Args:
            parent_values (list): List of numpy.ndarray resulting from the computation of the
            parent nodes

        Raises:
            AssertionError if the dimensions that were unknown at graph definition (generally,
            the first None dimensions) are compatible with the node functionality

        """
        pass

    @abstractmethod
    def _get_shape(self, parents):
        """Check that parents output shape is compatible with this node and compute the resulting
        output shape of this node. This check is computed at graph definition, when first
        dimensions can be None. Therefore, the resulting shape can have a first None dimension.

        Args:
            parents (list): List of parent ComputationNode that this node uses as input.

        Returns:
            The output shape of this node applied to the given parents.

        Raises:
            AssertionError if parent shapes are not compatible with the node implementation.

        """
        pass

    def __str__(self):
        return str(type(self)) + ', name: ' + self.name + ', out_shape: ' + str(self.shape)


class Values(ComputationNode):
    """Extends the ComputationNode to make it able to keep numerical values. It does not perform
    operations and cannot be called on other nodes, but can be trained and other nodes can be
    called on this.

    Attributes:
        values (numpy.ndarray) Numpy array containing the node values. Must be set by calling
            self.set_values(values) to ensure shape consistency.
        trainable (bool): Whether the values can be updated during training or must be kept fixed.
    """

    def __init__(self, shape, trainable=True, name=None):
        """Extends the ComputationNode constructor by setting the node shape on instantiation.

        Args:
            shape (tuple): Shape of the node, can have None in the first dimension.
            trainable (bool): If true the values can be updated by backpropagation.
        """
        super(Values, self).__init__(name=name)
        self.shape = shape if isinstance(shape, tuple) else (shape, )
        self.trainable = trainable
        self.values = None

    def set_values(self, values):
        """Set the given values in the node

        Args:
            values (numpy.ndarray): Numpy array consistent with self.shape.

        """
        shape_error_msg = 'Values shape ' + str(values.shape) + 'does not match with node shape '\
                                                              + str(self.shape)
        if self.shape[0] is None:
            assert self.shape[1:] == values.shape[1:], shape_error_msg
        else:
            assert self.shape == values.shape, shape_error_msg
        self.values = values

    def __call__(self, parents):
        """Values node has fixed shape and cannot have parents, therefore it cannot be called on
        other nodes.
        """
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
    TODO: Extend to any number of input nodes
    """

    def _eval_node(self, parents_values):
        """Return the dot product of the two input nodes.
        TODO: Make it consistent with any shape using np.einsum or np.matmul and np.squeeze

        Args:
            parents_values (list): List of numpy.ndarray of len = 2

        Returns:
            The dot product of the two inputs.

        """
        return np.dot(parents_values[0], parents_values[1])

    def _get_shape(self, parents):
        """Returns the shape of dot product result taking into account possible None dimensions
        of the inputs.

        Args:
            parents (list): List of two ComputationNode of compatible shapes for the dot product
            operation.

        Returns:
            The shape of the dot product of the two input nodes.

        Raises:
            AssertionError if parent shapes are not compatible.

        """
        assert len(parents) == 2
        res_shape = tuple()
        if parents[0].shape[0] is None or parents[1].shape[0] is None:
            res_shape += (None,)

        # Working now with shapes without initial None
        if parents[0].shape[0] is None:
            a_shape = parents[0].shape[1:]
        else:
            a_shape = parents[0].shape
        if parents[1].shape[0] is None:
            b_shape = parents[1].shape[1:]
        else:
            b_shape = parents[1].shape

        if len(a_shape) == 1:
            if len(b_shape) == 1:
                assert a_shape[0] == b_shape[0]
                return res_shape + (1,)
            else:
                assert a_shape[0] == b_shape[-2]
                return res_shape + b_shape[:-2] + (b_shape[-1],)
        else:  # len(a_shape) > 1
            if len(b_shape) == 1:
                assert a_shape[-1] == b_shape[0]
                return a_shape[:-1]
            else:  # Both shapes have len > 1
                assert a_shape[-1] == b_shape[-2]
                return a_shape[:-1] + b_shape[:-2] + (b_shape[-1],)

    def _check_eval_input(self, parent_values):
        """Check the dimensions of the input values that were None at graph definition

        Args:
            parent_values (list): List of numpy.ndarray with input values for the dot product

        Raises:
            AssertionError if both inputs had None first dimension in the definition and those
            dimensions do not match at runtime.

        """
        a_shape = self.parents[0].shape
        b_shape = self.parents[1].shape
        # Check runtime shapes
        if a_shape[0] is None and b_shape[0] is None:
            assert parent_values[0].shape[0] == parent_values[1].shape[0]


class Sum(ComputationNode):
    """Performs the sum operation between two parent nodes. The sum is performed element wise or by
    broadcasting depending on the parent node shapes. Works for only two inputs.
    TODO: Make it work for an arbitrary number of inputs and do appropriate shape check
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
        assert len(parents) == 1, 'Relu can be only called on one node'
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
        """Returns a the input flattened in a row vector, or parent_values.shape[0] row vectors
        if the first dimension was None.

        Args:
            parents_values (list): List with a single ComputationNode that has to be flattened.

        Returns:
            numpy.ndarray flattened to a row vector or an array of row vectors depending on
            whether the shape was defined with a None first dimension.

        """
        p = parents_values[0]
        if self.parents[0].shape[0] is None:
            return np.reshape(p, newshape=(p.shape[0], np.prod(p.shape[1:])))
        else:
            return np.reshape(p, newshape=(1, np.prod(p.shape),))

    def _check_eval_input(self, parent_values):
        pass  # Nothing to be checked here

    def _get_shape(self, parents):
        """Returns the shape of the Flatten node applied to the given parent.

        Args:
            parents (list): List of a single ComputationNode

        Returns:


        """
        assert len(parents) == 1, 'Flatten can be called only on one node'
        if parents[0].shape[0] is None:
            return None, np.prod(parents[0].shape[1:])
        else:
            return 1, np.prod(parents[0].shape)


if __name__ == '__main__':
    X = Values(shape=(None, 10))
    X.set_values(np.ones((15, 10)))

    W = Values(shape=(10, 3))  # E.g. 3 units layer
    W.set_values(np.zeros(W.shape))

    dot = DotProduct()([X, W])
    res_dot = dot.eval()

    print('Multiplying ' + str(X.shape) + ' with ' + str(W.shape))
    print('Actual values ' + str(X.values.shape) + ' and ' + str(W.values.shape))
    print('Result graph' + str(dot.shape) + ',  result actual ' + str(res_dot.shape))
    print('--------------------------------')

    b = Values(shape=(3,))
    b.set_values(np.ones((3,)))

    sum = Sum()([dot, b])
    res_sum = sum.eval()
    print('Summing ' + str(dot.shape) + ' with ' + str(b.shape))
    print('Actual values ' + str(res_dot.shape) + ' and ' + str(b.values.shape))
    print('Result graph ' + str(sum.shape) + ',  result actual ' + str(res_sum.shape))

    softmax = Softmax()(sum)
    res_soft = softmax.eval()
    print('Softmax on ' + str(sum.shape) + ', actual ' + str(res_sum.shape))
    print('Result graph ' + str(softmax.shape) + ',  result actual ' + str(res_soft.shape))
