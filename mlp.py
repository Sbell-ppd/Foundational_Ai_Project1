from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    n_samples = train_x.shape[0]
    for i in range(0, n_samples, batch_size):
        batch_x = train_x[i:i + batch_size]
        batch_y = train_y[i:i + batch_size]
        yield batch_x, batch_y
    pass


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)
    pass


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    pass


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)
    pass


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
     e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
     return e_x / np.sum(e_x, axis=-1, keepdims=True)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)
    pass


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    pass


class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    pass


class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sp = np.log(1 + np.exp(x))
        return np.tanh(sp) + x * (1 - np.tanh(sp) ** 2) * (1 / (1 + np.exp(-x)))
    pass



class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((y_true - y_pred) ** 2)
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
    pass


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.sum(y_true * np.log(y_pred + 1e-9), axis=1)
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
    pass


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction,dropuout_rate: float=0.0):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropuout_rate

         # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biaes
        self.W = np.random.randn(fan_in, fan_out) * 0.01 # weights
        self.b = np.zeros((1, fan_out)) # biases

    def forward(self, h: np.ndarray, training: bool=True):
        z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(z)
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, self.activations.shape) / (1 - self.dropout_rate)
            self.activations *= self.dropout_mask
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dropout_rate > 0.0:
            delta *= self.dropout_mask

        dL_dPhi = delta * self.activation_function.derivative(self.activations)
        dl_dw = np.dot(h.T, dL_dPhi)
        dl_db = np.sum(dL_dPhi, axis=0, keepdims=True)
        self.delta = np.dot(dL_dPhi, self.W.T)
        return dl_dw, dl_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h = input_data if i == 0 else self.layers[i - 1].activations
            dl_dw, dl_db = layer.backward(h, delta)
            dl_dw_all.insert(0, dl_dw)
            dl_db_all.insert(0, dl_db)
            delta = layer.delta
        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, rmsprop:bool=False, beta: float=0.9, epsilon:float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
        training_losses = []
        validation_losses = []
        #initialize cache for RMSprop
        if rmsprop:
            cache_W = [np.zeros_like(layer.W) for layer in self.layers]
            cache_b = [np.zeros_like(layer.b) for layer in self.layers]
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                y_pred = self.forward(batch_x)
                loss = loss_func.loss(batch_y, y_pred)
                epoch_loss += np.mean(loss)
                loss_grad = loss_func.derivative(batch_y, y_pred)
                dl_dw_all, dl_db_all = self.backward(loss_grad, batch_x)
                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        #update cache
                        cache_W[i] = beta * cache_W[i] + (1 - beta) * dl_dw_all[i] ** 2
                        cache_b[i] = beta * cache_b[i] + (1 - beta) * dl_db_all[i] ** 2
                        #update weights and biases using RMSprop
                        layer.W -= learning_rate * dl_dw_all[i] / (np.sqrt(cache_W[i]) + epsilon)
                        layer.b -= learning_rate * dl_db_all[i] / (np.sqrt(cache_b[i]) + epsilon)
                    else:
                        #vanilla SGD update
                        layer.W -= learning_rate * dl_dw_all[i]
                        layer.b -= learning_rate * dl_db_all[i]
            training_losses.append(epoch_loss/len(train_x))
            val_pred = self.forward(val_x)
            val_loss = np.mean(loss_func.loss(val_y, val_pred))
            validation_losses.append(val_loss)
        return np.array(training_losses), np.array(validation_losses)


class CrossEntropyLoss:
    def loss(self,y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-9
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)]+epsilon)
        loss = np.sum(log_likelihood) / m
        return loss
    def derivative(self, y_true, y_pred):
        m = y_true.shape[0]
        grad = y_pred.copy()
        grad[range(m), y_true.argmax(axis=1)] -= 1
        grad /= m
        return grad

    pass