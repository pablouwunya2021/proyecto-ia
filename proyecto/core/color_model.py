import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ColorMLP:

    def __init__(self):

        # ---- Mapeo color → clase ----
        self.label_to_class = {
            "Green": 0,
            "Blue": 1,
            "Grey": 2,
            "Yellow": 3,
            "Brown": 4,
            "Pink": 5,
            "Orange": 6,
            "Purple": 7,
            "Red": 8,
            "White": 9,
            "Black": 10
        }

        self.class_to_label = {v: k for k, v in self.label_to_class.items()}

        # ---- Mapeo color → costo ----
        self.label_to_cost = {
            "Green": 3,
            "Blue": 10,
            "Grey": 1,
            "Yellow": 5,
            "Brown": 4,
            "Pink": 6,
            "Orange": 7,
            "Purple": 8,
            "Red": 5,
            "White": 2,
            "Black": 2
        }

        self.input_size = 3
        self.hidden1 = 16
        self.hidden2 = 16
        self.output_size = len(self.label_to_class)

        np.random.seed(42)

        self.W1 = np.random.randn(self.input_size, self.hidden1) * 0.01
        self.b1 = np.zeros((1, self.hidden1))

        self.W2 = np.random.randn(self.hidden1, self.hidden2) * 0.01
        self.b2 = np.zeros((1, self.hidden2))

        self.W3 = np.random.randn(self.hidden2, self.output_size) * 0.01
        self.b3 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def forward(self, X):

        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        a3 = self.softmax(z3)

        return z1, a1, z2, a2, z3, a3


    def backward(self, X, y_true, cache, lr):

        z1, a1, z2, a2, z3, a3 = cache
        m = X.shape[0]

        dz3 = a3 - y_true
        dW3 = a2.T @ dz3 / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = (dz3 @ self.W3.T) * self.relu_derivative(z2)
        dW2 = a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * self.relu_derivative(z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3


    def one_hot(self, y):
        oh = np.zeros((len(y), self.output_size))
        oh[np.arange(len(y)), y] = 1
        return oh


    def train(self, csv_path, epochs=200, lr=0.05, batch_size=32):

        data = pd.read_csv(csv_path)

        X = data[['red','green','blue']].values / 255.0
        y_labels = data['label'].values

        y = np.array([self.label_to_class[label] for label in y_labels])
        y_encoded = self.one_hot(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        for epoch in range(epochs):

            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                cache = self.forward(X_batch)
                self.backward(X_batch, y_batch, cache, lr)

            if epoch % 20 == 0:
                acc = self.evaluate(X_test, y_test)
                print(f"Epoch {epoch} - Test Accuracy: {acc:.4f}")

        print("Final Accuracy:", self.evaluate(X_test, y_test))


    def evaluate(self, X, y_true):
        _,_,_,_,_,y_pred = self.forward(X)
        pred = np.argmax(y_pred, axis=1)
        true = np.argmax(y_true, axis=1)
        return np.mean(pred == true)


    def predict_cost(self, rgb):

        rgb = np.array(rgb).reshape(1,3) / 255.0
        _,_,_,_,_,pred = self.forward(rgb)

        class_id = np.argmax(pred)
        label = self.class_to_label[class_id]

        return self.label_to_cost[label]
