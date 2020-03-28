import numpy as np
import computation_graph
from abc import ABC, abstractmethod
from computation_graph import DotProduct, Multiply, Log, ReduceSum, Variable


class UnknownLossException(Exception):
    pass


class Loss(ABC):
    """Loss is an abstract class that represent a loss function. Class that extend Loss should
    implement the part of the computational graph that is used to compute the loss during training.
    """

    @abstractmethod
    def __call__(self, predictions, labels):
        """Create the ComputationNode objects to compute the loss function between the model
        predictions and true labels.

        A Loss object must be called on two ComputationNode objects: the prediction node and
        the labels node. The prediction node is the output node of the network, and the labels
        node is the placeholder node for the true labels.

        Args:
            predictions (ComputationNode): Output node for which the loss is to be computed,
                of shape (batch_size, d0, d1, ..., dN)
            labels (ComputationNode): The ground truth node of shape (batch_size, d0, d1, ..., dn)

        Returns:
            The output ComputationNode of the graph block representing the loss function.

        """


class CategoricalCrossentropy(Loss):

    def __call__(self, predictions, labels):
        # No need to transpose since both have first dimension == None
        dot = DotProduct([predictions, labels])
        log = Log()(dot)
        minus_one = Variable(-1., trainable=False)
        neg_log = Multiply(log, minus_one)
        reduce_sum = ReduceSum()(neg_log)
        return reduce_sum


losses_dict = {'categorical-crossentropy': CategoricalCrossentropy}


def get_loss(name):
    try:
        return losses_dict[name]
    except KeyError:
        valid_losses_str = ', '.join(losses_dict.keys())
        raise UnknownLossException('Loss ' + name + ' is not known. Valid losses are ' +
                                   valid_losses_str)
