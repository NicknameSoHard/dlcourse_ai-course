import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.trace(np.matmul(W.T, W))   # L2(W) = λ * tr(W.T * W)
    grad = 2 * reg_strength * W                         # dL2(W)/dW = 2 * λ * W
    return loss, grad   # L2(W), dL2(W)/dW


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) - classifier output

    Returns:
      probs, np array of the same shape as predictions - probability for every class, 0..1
    """

    maximums = np.amax(predictions, axis=1).reshape(predictions.shape[0], 1)
    predictions_ts = predictions - maximums

    predictions_exp = np.exp(predictions_ts)
    sums = np.sum(predictions_exp, axis=1).reshape(predictions_exp.shape[0], 1)
    result = predictions_exp / sums

    return result   # S


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) - probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)

    Returns:
      loss: single value
    """

    rows = np.arange(target_index.shape[0])
    cols = target_index

    return np.mean(-np.log(probs[rows, cols]))  # L


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions, including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) - classifier output
      target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    probs = softmax(predictions)                    # S
    loss = cross_entropy_loss(probs, target_index)  # L

    indicator = np.zeros(probs.shape)
    indicator[np.arange(probs.shape[0]), target_index] = 1      # 1(y)
    dprediction = (probs - indicator) / predictions.shape[0]    # dL/dZ = (S - 1(y)) / N

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    def reset_grad(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        self.d_out_result = None

    def forward(self, X):
        self.d_out_result = np.greater(X, 0).astype(float)  # dZ/dX
        return np.maximum(X, 0)                             # Z

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = np.multiply(d_out, self.d_out_result)    # dL/dX = dL/dZ * dZ/dX
        return d_result     # dL/dX

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass:
        computes gradient with respect to input and accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient with respect to input
        """

        d_result = np.matmul(d_out, self.W.value.T)     # dL/dX = dL/dZ * dZ/dX = dL/dZ * W.T
        dLdW = np.matmul(self.X.T, d_out)               # dL/dW = dL/dZ * dZ/dW = X.T * dL/dZ
        dLdB = 2 * np.mean(d_out, axis=0)               # dL/dW = dL/dZ * dZ/dB = I * dL/dZ

        self.W.grad += dLdW
        self.B.grad += dLdB

        return d_result     # dL/dX

    def params(self):
        return {'W': self.W, 'B': self.B}
