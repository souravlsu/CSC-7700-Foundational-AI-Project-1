import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple

# -------------------- Activation Functions -------------------- #
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """        
        pass

# Different activation functions used in neural networks
class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)

class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)

class Softmax(ActivationFunction):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def derivative(self, x):
        softmax_x = self.forward(x)
        # Compute the Jacobian matrix for each input
        jacobian = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        
        for i in range(x.shape[0]):
            s = softmax_x[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(s) - np.dot(s, s.T)
        return jacobian

class Linear(ActivationFunction):
    def forward(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))
    
    def derivative(self, x):
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def derivative(self, x):
        omega = np.exp(3*x) + 4*np.exp(2*x) + (6+4*x)*np.exp(x) + 4*(1 + x)
        delta = 1 + pow((np.exp(x) + 1), 2)
        return np.exp(x) * omega / pow(delta, 2)

# -------------------- Loss Functions -------------------- #
class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass
    
    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass

# Mean Squared Error (MSE) Loss
class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
# Cross-Entropy Loss for classification
class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-9, 1 - 1e-9)), axis=1))  # Clip values

    def derivative(self, y_true, y_pred):
        return y_pred - y_true  # Correct gradient for softmax + cross-entropy


# -------------------- Layer Class -------------------- #
# Represents a fully connected layer with dropout and activation function
class Layer:
    def __init__(self, fan_in, fan_out, activation_function, dropout_rate=0.0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        limit = np.sqrt(6 / (fan_in + fan_out))     # Xavier/Glorot Initialization
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out))
        self.dropout_mask = None

    def forward(self, h, training=True):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.h_input = h
        self.z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(self.z)
        
        # Apply dropout if training
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = (np.random.rand(*self.activations.shape) > self.dropout_rate).astype(float)
            self.activations *= self.dropout_mask
            self.activations /= (1.0 - self.dropout_rate)
        
        return self.activations

    def backward(self, delta):
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        # If activation is Softmax, delta already includes derivative (from CrossEntropy derivative)
        if isinstance(self.activation_function, Softmax):
            dL_dZ = delta  # Softmax derivative is included in loss gradient
        else:
            dL_dZ = delta * self.activation_function.derivative(self.z)

        if self.dropout_rate > 0.0 and self.dropout_mask is not None:
            dL_dZ *= self.dropout_mask
            dL_dZ /= (1.0 - self.dropout_rate)

        dL_dW = np.dot(self.h_input.T, dL_dZ)
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)
        dL_dH = np.dot(dL_dZ, self.W.T)
        
        return dL_dW, dL_db, dL_dH

# -------------------- Multilayer Perceptron -------------------- #
class MultilayerPerceptron:
    def __init__(self, layers):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x, training=True):
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, loss_grad):
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        gradients = []
        delta = loss_grad
        
        for layer in reversed(self.layers):
            dW, db, delta = layer.backward(delta)
            gradients.append((dW, db))
        
        return gradients[::-1]  # Reverse to match layer order

    def train(self, train_x, train_y, val_x, val_y, loss_func, learning_rate=1E-3, batch_size=16, epochs=32, optimizer='vanilla', beta=0.9, epsilon=1e-8, momentum=0.9):
        """
        Train the multilayer perceptron with momentum applied to gradient updates.

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :param optimizer: optimization method ('vanilla' or 'rmsprop')
        :param beta: parameter for RMSProp
        :param epsilon: small value to prevent division by zero
        :param momentum: momentum coefficient applied to weight updates
        :return: training and validation losses
        """
        training_losses, validation_losses = [], []

        # Initialize velocity terms for momentum
        velocity = [{"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)} for layer in self.layers]

        if optimizer == 'rmsprop':
            caches = [{"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)} for layer in self.layers]

        for epoch in range(epochs):
            indices = np.arange(train_x.shape[0])
            np.random.shuffle(indices)
            train_x, train_y = train_x[indices], train_y[indices]

            batch_losses = []
            for i in range(0, train_x.shape[0], batch_size):
                batch_x, batch_y = train_x[i:i+batch_size], train_y[i:i+batch_size]
                y_pred = self.forward(batch_x)

                loss = loss_func.loss(batch_y, y_pred)
                batch_losses.append(loss)

                gradients = self.backward(loss_func.derivative(batch_y, y_pred))
                for idx, (layer, (dW, db)) in enumerate(zip(self.layers, gradients)):
                    # Apply momentum to gradient updates
                    velocity[idx]["W"] = momentum * velocity[idx]["W"] - learning_rate * dW
                    velocity[idx]["b"] = momentum * velocity[idx]["b"] - learning_rate * db

                    if optimizer == 'vanilla':
                        layer.W += velocity[idx]["W"]
                        layer.b += velocity[idx]["b"]
                    elif optimizer == 'rmsprop':
                        caches[idx]["W"] = beta * caches[idx]["W"] + (1 - beta) * (dW ** 2)
                        caches[idx]["b"] = beta * caches[idx]["b"] + (1 - beta) * (db ** 2)

                        layer.W += velocity[idx]["W"] / (np.sqrt(caches[idx]["W"]) + epsilon)
                        layer.b += velocity[idx]["b"] / (np.sqrt(caches[idx]["b"]) + epsilon)

            training_losses.append(np.mean(batch_losses))
            val_loss = loss_func.loss(val_y, self.forward(val_x))
            validation_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {training_losses[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}")

        return training_losses, validation_losses


