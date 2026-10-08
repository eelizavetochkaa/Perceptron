import numpy as np


class MLP:
    def __init__(self, input_size=30, hidden_sizes=(32, 16), output_size=2, seed=42):
        rng = np.random.default_rng(seed)
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2 / sizes[i]) if i < len(sizes) - 2 else np.sqrt(1 / sizes[i])
            self.weights.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.biases.append(np.zeros((1, sizes[i + 1])))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            zs.append(z)
            activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        return self.softmax(z), (activations, zs)

    @staticmethod
    def cross_entropy(y_true, probabilities):
        eps = 1e-12
        return -np.mean(np.sum(y_true * np.log(probabilities + eps), axis=1))

    def backward(self, cache, y_true):
        activations, zs = cache
        n = activations[0].shape[0]
        dW = [None] * len(self.weights)
        db = [None] * len(self.weights)

        dz = (self.softmax(zs[-1]) - y_true) / n
        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = activations[i].T @ dz
            db[i] = np.sum(dz, axis=0, keepdims=True)
            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.relu_derivative(zs[i - 1])

        return dW, db

    def train_step(self, X, y, learning_rate):
        probabilities, cache = self.forward(X)
        dW, db = self.backward(cache, y)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
        return self.cross_entropy(y, probabilities)

    def predict_proba(self, X):
        probabilities, _ = self.forward(X)
        return probabilities

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path):
        params = {}
        for i in range(len(self.weights)):
            params[f'W{i + 1}'] = self.weights[i]
            params[f'b{i + 1}'] = self.biases[i]
        np.savez(path, **params)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        n_layers = len([k for k in data.files if k.startswith('W')])
        weights = [data[f'W{i + 1}'] for i in range(n_layers)]
        biases = [data[f'b{i + 1}'] for i in range(n_layers)]
        model = cls(input_size=weights[0].shape[0],
                    hidden_sizes=[w.shape[1] for w in weights[:-1]],
                    output_size=weights[-1].shape[1])
        model.weights = weights
        model.biases = biases
        return model