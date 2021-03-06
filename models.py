from computation_graph import Values, Sum
from layers import Dense
import numpy as np
from losses import get_loss


class Model:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = None  # Placeholder for labels in training
        self.loss_output = None  # Loss output node

    def predict(self, input_values):
        if isinstance(self.inputs, list):
            assert len(self.inputs) == len(input_values)

        self.inputs.set_values(input_values)
        return self.outputs.eval()

    def __call__(self, input_values):
        return self.predict(input_values)

    def compile(self, loss):
        """Compile the model by applying the given loss to the computation graph.

        Args:
            loss (str): Name of the loss function. Loss functions are defined in losses module.

        Raises:
            UnknownLossException if the given loss name does not correspond to a loss function
            implemented in the losses module.

        """
        self.labels = Values(shape=self.outputs.shape, trainable=False)
        self.loss_output = get_loss(loss)(self.outputs, self.labels)


if __name__ == '__main__':
    x = Values(shape=(None, 5))
    k = Dense(units=16)(x)
    model = Model(x, k)
    model.compile()
    res = model(np.ones(shape=(20, 5)))
