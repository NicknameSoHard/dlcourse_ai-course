import numpy as np

from layers import FullyConnectedLayer, ReLULayer, ConvolutionalLayer, \
    MaxPoolingLayer, Flattener, softmax_with_cross_entropy, l2_regularization


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.model = [
            ConvolutionalLayer(in_channels=input_shape[2],
                               out_channels=conv1_channels,
                               filter_size=3,
                               padding=1),
            ReLULayer(),
            MaxPoolingLayer(2, 2),
            ConvolutionalLayer(in_channels=conv1_channels,
                               out_channels=conv2_channels,
                               filter_size=3,
                               padding=1),
            ReLULayer(),
            MaxPoolingLayer(2, 2),
            Flattener(),
            FullyConnectedLayer(n_input=int(input_shape[0] / 4) ** 2 * conv2_channels,
                                n_output=n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for layer in self.model:
            layer.reset_grad()

        X_ = X.copy()
        for layer in self.model:
            X_ = layer.forward(X_)

        loss, d_pred = softmax_with_cross_entropy(X_, y)

        d_out = d_pred.copy()
        for layer in reversed(self.model):
            d_out = layer.backward(d_out)

        return loss

    def predict(self, X):
        pred = X
        for layer in self.model:
            pred = layer.forward(pred)

        return pred.argmax(axis=1)

    def params(self):
        result = {}

        for num_layer, layer in enumerate(self.model):
            for key, value in layer.params().items():
                result[(num_layer, key)] = value

        return result