import sys
sys.path.append('../')

from assignment2.layers import softmax, cross_entropy_loss

import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.trace(np.matmul(W.T, W))   # L2(W) = λ * tr(W.T * W)
    grad = 2 * reg_strength * W                         # dL2(W)/dW = 2 * λ * W

    return loss, grad   # L2(W), dL2(W)/dW


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    if len(probs.shape) == 1:
        subtr = np.zeros(probs.shape)
        subtr[target_index] = 1
        dprediction = probs - subtr
    else:
        subtr = np.zeros(probs.shape)
        subtr[range(len(target_index)), target_index] = 1
        dprediction = (probs - subtr) / (predictions.shape[0])

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        layer_X = X.copy()
        layer_X[layer_X < 0] = 0
        return layer_X

    def backward(self, d_out):
        X_back = self.X.copy()
        X_back[X_back > 0] = 1
        X_back[X_back <= 0] = 0
        d_result = X_back * d_out
        return d_result

    def params(self):
        return {}

    def reset_grad(self):
        pass


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = Param(X.copy())
        output = np.dot(self.X.value, self.W.value) + self.B.value
        return output

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.value.T, d_out)
        self.B.grad = np.array([np.sum(d_out, axis=0)])
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X

        if self.padding:
            self.X = np.zeros((batch_size,
                               height + 2 * self.padding,
                               width + 2 * self.padding,
                               channels), dtype=X.dtype)
            self.X[:, self.padding: -self.padding, self.padding: -self.padding, :] = X

        _, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        output = []

        for y in range(out_height):
            row = []
            for x in range(out_width):
                x_window = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]
                x_window = np.transpose(x_window, axes=[0, 3, 2, 1])
                x_window = x_window.reshape((batch_size, self.filter_size ** 2 * channels))

                w_window = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                w_reshape = w_window.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels))
                out = np.dot(x_window, w_reshape)

                row.append(np.array([out], dtype=self.W.value.dtype).reshape((batch_size, 1, 1, self.out_channels)))

            output.append(np.dstack(row))

        output = np.hstack(output) + self.B.value
        return output

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_inp = np.zeros(self.X.shape)
        for y in range(out_height):
            for x in range(out_width):
                d_window = d_out[:, y, x, :]

                x_window = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]

                x_window = np.transpose(x_window, axes=[0, 3, 1, 2])
                x_window = x_window.reshape((batch_size, self.filter_size ** 2 * channels))
                x_transpose = x_window.transpose()

                w_window = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                w_window = w_window.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels))
                w_transpose = w_window.transpose()

                d_w_window = np.dot(x_transpose, d_window)
                d_w_window = d_w_window.reshape(self.in_channels, self.filter_size, self.filter_size, self.out_channels)
                d_w_transpose = np.transpose(d_w_window, axes=[2, 1, 0, 3])

                self.W.grad += d_w_transpose
                E = np.ones(shape=(1, batch_size))
                B = np.dot(E, d_window)
                B = B.reshape((d_window.shape[1]))
                self.B.grad += B

                d_inp_xy = np.dot(d_window, w_transpose)#d_window.dot(w_window.transpose())
                d_inp_xy = d_inp_xy.reshape((batch_size, channels, self.filter_size, self.filter_size))

                d_inp_xy = np.transpose(d_inp_xy, axes=[0, 3, 2, 1])

                d_inp[:, y: y + self.filter_size, x: x + self.filter_size, :] += d_inp_xy

        if self.padding:
            d_inp = d_inp[:, self.padding: -self.padding, self.padding: -self.padding, :]

        return d_inp

    def params(self):
        return {'W': self.W, 'B': self.B }

    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X

        out_height = (height - self.pool_size) / self.stride + 1
        out_width = (width - self.pool_size) / self.stride + 1

        if (not float(out_height).is_integer() and not float(out_width).is_integer()):
            raise Exception(f"Stride and pool size aren't consistent for {height}, {width}")

        out = np.zeros([int(batch_size), int(out_height), int(out_width), int(channels)])
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        y_1 = 0
        for y in range(int(out_height)):
            x_1 = 0
            for x in range(int(out_width)):
                out[:, y, x, :] += np.amax(self.X[:, y_1:y_1 + self.pool_size, x_1:x_1 + self.pool_size, :],
                                           axis=(1, 2))
                x_1 += self.stride
            y_1 += self.stride
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, channels = d_out.shape
        in_l = np.zeros_like(self.X)

        for b in range(batch_size):
            for ch in range(channels):
                y_1 = 0
                for y in range(out_height):
                    x_1 = 0
                    for x in range(out_width):
                        ind = np.unravel_index(
                            np.argmax(self.X[b, y_1:y_1 + self.pool_size, x_1:x_1 + self.pool_size, ch]),
                            self.X[b, y_1:y_1 + self.pool_size, x_1:x_1 + self.pool_size, ch].shape)
                        in_l[b, y_1:y_1 + self.pool_size, x_1:x_1 + self.pool_size, ch][ind[0], ind[1]] = d_out[
                            b, y, x, ch]
                        x_1 += self.stride
                    y_1 += self.stride
        return in_l

    def params(self):
        return {}

    def reset_grad(self):
        pass


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = batch_size, height, width, channels
        x_reshaped = X.reshape(batch_size, height * width * channels)
        return x_reshaped

    def backward(self, d_out):
        reshaped_out = d_out.reshape(self.X_shape[0], self.X_shape[1], self.X_shape[2], self.X_shape[3])
        return reshaped_out

    def params(self):
        # No params!
        return {}

    def reset_grad(self):
        pass
