"""This module provides initializer function for ComputationNode and Layer weights
"""

import numpy as np


class UnknownInitializerException(Exception):
    pass


def get_initializer(name):
    """Returns an initializer function for the given name.

    The initializer function takes a computation_graph.Values object as input and performs weight
    initialization.

    Args:
        name (str): Name of the initializer ('he', xavier', 'zero')

    Returns:
        initializer function of type func(Values)

    """
    try:
        initializers[name]
    except KeyError:
        raise UnknownInitializerException('Initializer ' + str(name) + ' does not exist.')


def he_initializer(weights):
    stdev = np.sqrt(2. / weights.shape[-1])
    weights.set_values(np.random.standard_normal(size=weights.shape) / stdev)


def xavier_initializer(weights):
    raise NotImplementedError


def zero_initializer(weights):
    weights.set_values(np.zeros(weights.shape))


initializers = {'he': he_initializer,
                'xavier': xavier_initializer,
                'zero': zero_initializer}

