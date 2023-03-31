from typing import List, Protocol, Tuple, Callable
from dataclasses import dataclass, field
from timeit import timeit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


GradientTape = List[List[np.ndarray]]
Optimizer = Callable[[GradientTape], GradientTape]
DatasetSplits = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class Layer(Protocol):
    def compile(self, input_dims: int):
        raise NotImplementedError()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(
            self, orig_inputs: np.ndarray, orig_outputs: np.ndarray,
            delta: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    def update(self, grads: List[np.ndarray]):
        raise NotImplementedError()


class Loss(Protocol):
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError()

    def loss_delta(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class Model:
    layers: List[Layer]
    outputs_cache: List[np.ndarray] = field(default_factory=list)

    def compile(self, input_dims: int):
        self.outputs_cache.append([])
        for layer in self.layers:
            layer.compile(input_dims)
            self.outputs_cache.append([])
            if isinstance(layer, DenseLayer):
                input_dims = layer.output_dims

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs_cache[0] = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer.forward(inputs)
            self.outputs_cache[i+1] = outputs
            inputs = outputs
        return outputs

    def backward(self, delta: np.ndarray) -> Tuple[GradientTape, np.ndarray]:
        grads_tape: GradientTape = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            inputs, outputs = self.outputs_cache[i], self.outputs_cache[i+1]
            grads, delta = layer.backward(inputs, outputs, delta)
            grads_tape.insert(0, grads)

        return grads_tape, delta

    def update(self, grads: GradientTape):
        for layer_grads, layer in zip(grads, self.layers):
            layer.update(layer_grads)


class ReLULayer:
    def compile(self, input_dims: int):
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0.0, inputs, 0.0)

    def backward(
            self, orig_inputs: np.ndarray, orig_outputs: np.ndarray,
            delta: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        return [], np.where(orig_inputs > 0.0, delta, 0.0)

    def update(self, grads: List[np.ndarray]):
        pass


class SigmoidLayer:
    def compile(self, input_dims: int):
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(inputs * -1.0))

    def backward(
            self, orig_inputs: np.ndarray, orig_outputs: np.ndarray,
            delta: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        return [], orig_outputs * (1.0 - orig_outputs) * delta

    def update(self, grads: List[np.ndarray]):
        pass


@dataclass
class DenseLayer:
    output_dims: int
    weights: np.ndarray = field(init=False)
    biases: np.ndarray = field(init=False)

    def compile(self, input_dims: int):
        self.weights = np.random.normal(0.0, 0.1, (input_dims, self.output_dims))
        self.biases = np.zeros((1, self.output_dims))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.matmul(inputs, self.weights) + self.biases

    def backward(
            self, orig_inputs: np.ndarray, orig_outputs: np.ndarray,
            delta: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        weight_grads = np.matmul(np.transpose(orig_inputs), delta)
        bias_grads = np.sum(delta, axis=0)
        delta = np.matmul(delta, np.transpose(self.weights))
        return [weight_grads, bias_grads], delta

    def update(self, grads: List[np.ndarray]):
        weight_grads, bias_grads = grads
        self.weights -= weight_grads
        self.biases -= bias_grads


class MSELoss:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.power(y_pred - y_true, 2))

    def loss_delta(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size


@dataclass
class NaiveSGDOptimizer:
    learn_rate: float = 0.01

    def __call__(self, grad_tape: GradientTape) -> GradientTape:
        return [[g * self.learn_rate for g in layer_grads] for layer_grads in grad_tape]


@dataclass
class MomentumSGDOptimizer:
    learn_rate: float = 0.001
    gamma: float = 0.9
    v: GradientTape = field(init=False, default=None)

    def __call__(self, grad_tape: GradientTape) -> GradientTape:
        if not self.v:
            self._init_moment_vectors(grad_tape)

        for i, (layer_grads, layer_v_prev) in enumerate(zip(grad_tape, self.v)):
            for j, (grads, v_prev) in enumerate(zip(layer_grads, layer_v_prev)):
                adj_grads = v_prev * self.gamma + self.learn_rate * grads
                grad_tape[i][j] = adj_grads
                self.v[i][j] = adj_grads

        return grad_tape

    def _init_moment_vectors(self, grad_tape: GradientTape):
        self.v = [[np.zeros_like(g) for g in l] for l in grad_tape]


@dataclass
class AdamOptimizer:
    learn_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    t: int = 0
    m: GradientTape = field(init=False, default=None)
    v: GradientTape = field(init=False, default=None)

    def __call__(self, grad_tape: GradientTape) -> GradientTape:
        self.t += 1
        if not self.m:
            self._init_moment_vectors(grad_tape)

        for i, (layer_grads, layer_m_prev, layer_v_prev) in enumerate(zip(grad_tape, self.m, self.v)):
            for j, (grads, m_prev, v_prev) in enumerate(zip(layer_grads, layer_m_prev, layer_v_prev)):
                m_curr = self.beta_1 * m_prev + (1 - self.beta_1) * grads
                v_curr = self.beta_2 * v_prev + (1 - self.beta_2) * (grads * grads)
                self.m[i][j], self.v[i][j] = m_curr, v_curr
                m_curr_est = m_curr / (1 - np.power(self.beta_1, self.t))
                v_curr_est = v_curr / (1 - np.power(self.beta_2, self.t))
                adj_grads = self.learn_rate * m_curr_est / (np.sqrt(v_curr_est) + self.epsilon)
                grad_tape[i][j] = adj_grads

        return grad_tape

    def _init_moment_vectors(self, grad_tape: GradientTape):
        self.m, self.v = [], []
        for layer_grads in grad_tape:
            self.m.append([np.zeros_like(g) for g in layer_grads])
            self.v.append([np.zeros_like(g) for g in layer_grads])


def train(
        model: Model, dataset: DatasetSplits, loss: Loss,
        optimizer: Optimizer, epochs: int, batch_size: int):

    x_train, y_train, x_test, y_test = dataset
    num_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for i in range(num_batches):
            k, l = i * batch_size, (i + 1) * batch_size
            x, y_true = x_train[k:l], y_train[k:l]
            y_pred = model.forward(x)
            loss_delta = loss.loss_delta(y_pred, y_true)
            grads, _ = model.backward(loss_delta)
            grads = optimizer(grads)
            model.update(grads)

        test_pred = model.forward(x_test)
        print(f"epoch {epoch}, loss {loss.loss(test_pred, y_test)}", end="\r")


def partition_dataset(x: np.ndarray, y: np.ndarray) -> DatasetSplits:
    df = pd.DataFrame(np.arange(x.shape[0]))
    train_split = df.sample(frac=0.8, random_state=25)
    test_split = df.drop(train_split.index)
    x_train, y_train = x[train_split.index], y[train_split.index]
    x_test, y_test = x[test_split.index], y[test_split.index]
    return x_train, y_train, x_test, y_test


def create_model(feature_dims: int) -> Model:
    model = Model([
        DenseLayer(64),
        ReLULayer(),
        DenseLayer(64),
        ReLULayer(),
        DenseLayer(1)
    ])
    model.compile(feature_dims)
    return model


def plot_predictions(x, y_pred, y_true):
    plt.scatter(x, y_true)
    plt.scatter(x, y_pred)
    plt.show()


def main():
    x = np.expand_dims(np.linspace(-10, 10, 10_000), axis=-1)
    y = np.sin(x)
    data_splits = partition_dataset(x, y)
    x_train, _, x_test, y_test = data_splits

    feature_dims = x_train.shape[1]
    model = create_model(feature_dims)

    epochs = 10
    learn_rate = 0.01
    batch_size = 64

    optimizer = AdamOptimizer(learn_rate)
    loss = MSELoss()

    train_benchmark = lambda: train(
        model, data_splits, loss, optimizer, epochs, batch_size)

    repetitions = 10
    time_secs = timeit(train_benchmark, number=repetitions)
    print(f"time per training: {(time_secs / repetitions):.2f} s")

    # y_pred = model.forward(x_test)
    # plot_predictions(x_test, y_pred, y_test)


if __name__ == "__main__":
    main()
