from computation_graph import Values, Sum
from layers import Dense
import numpy as np


class Model:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, input_values):
        if isinstance(self.inputs, list):
            assert len(self.inputs) == len(input_values)

        self.inputs.set_values(input_values)
        return self.outputs.eval()

    def __call__(self, input_values):
        return self.predict(input_values)


if __name__ == '__main__':
    x = Values(shape=(None, 5))
    k = Dense(units=16)(x)
    model = Model(x, k)
    res = model(np.ones(shape=(20, 5)))
