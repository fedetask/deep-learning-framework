import computation_graph

activations_map = {'softmax': computation_graph.Softmax, 'relu': computation_graph.Relu}


def get_activation(activation):
    if activation is None:
        return None
    if isinstance(activation, str):
        return activations_map[activation]
    else:
        raise NotImplementedError('Only activation search by name is implemented')