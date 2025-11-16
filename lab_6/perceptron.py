import csv
import numpy as np
from sklearn.utils import shuffle

# READ DATA FROM CSV
def read_data(filepath):
    features = []
    labels = []
    class_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for entry in reader:
            sample = [
                float(entry['SepalLengthCm']),
                float(entry['SepalWidthCm']),
                float(entry['PetalLengthCm']),
                float(entry['PetalWidthCm'])
            ]
            label = class_map.get(entry['Species'])
            if label is not None:
                features.append(sample)
                labels.append(label)
    return np.array(features), np.array(labels)


# Basic perceptron (binary)
class BinaryPerceptron:
    def __init__(self, learning_rate=0.01, iteration=1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weight = None
        self.bias = None

    def train(self, data, targets):
        sample_no, feature_no = data.shape
        self.weight = np.zeros(feature_no)
        self.bias = 0
        for _ in range(self.iteration):
            for index, x_vec in enumerate(data):
                output = np.dot(x_vec, self.weight) + self.bias
                y_hat = self._activation(output)
                update = self.learning_rate * (targets[index] - y_hat)
                self.weight += update * x_vec
                self.bias += update

    def predict(self, data):
        outputs = np.dot(data, self.weight) + self.bias
        return np.where(outputs >= 0, 1, 0)


    def raw_output(self, data):
        return np.dot(data, self.weight) + self.bias

    def _activation(self, x):
        return np.where(x >= 0, 1, 0)


# Combined dual perceptron
class DualPerceptronModel:

    # first perceptron: tell if class 0 or not class 0
    # second perceptron: tell if it's class 1 or class 2
    
    # basic logic:
    # if type == 0:
    #     return 0
    # else:
    #     if type == 1:
    #         return 1
    #     else: 
    #         return 2
    
    # initialize the values and perceptrons
    def __init__(self, learning_rate=0.01, iteration=1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.first_perceptron = BinaryPerceptron(learning_rate, iteration)
        self.second_perceptron = BinaryPerceptron(learning_rate, iteration)


    def train(self, X, y):
        # Train first perceptron: type 0 or not type 0
        y_zero = np.where(y == 0, 1, 0)
        self.first_perceptron.train(X, y_zero)

        # Train second perceptron: type 1 or type 2
        mask = (y == 1) | (y == 2)
        X_sub = X[mask]
        y_sub = y[mask]
        y_bin = np.where(y_sub == 1, 1, 0)

        if np.unique(y_bin).size < 2:
            self.second_perceptron.weight = np.zeros(X.shape[1])
            self.second_perceptron.bias = 0
        else:
            self.second_perceptron.train(X_sub, y_bin)

    def predict(self, X):
        first_scores = self.first_perceptron.raw_output(X)
        is_zero = (first_scores >= 0)

        predictions = np.full(X.shape[0], 2, dtype=int)
        predictions[is_zero] = 0
        index_nonzero = np.where(~is_zero)[0]
        if index_nonzero.size > 0:
            second_scores = self.second_perceptron.raw_output(X[index_nonzero])
            predictions[index_nonzero] = np.where(second_scores >= 0, 1, 2)
        return predictions


class MultiClassPerceptron:
    def __init__(self, class_no, learning_rate=0.01, iteration=1000):
        self.class_no = class_no
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None  
        self.biases = None  

    def train(self, X, y):
        sample_no, feature_no = X.shape
        self.weights = np.zeros((self.class_no, feature_no))
        self.biases = np.zeros(self.class_no)

        for _ in range(self.iteration):
            for i in range(sample_no):
                x_i = X[i]
                correct_label = y[i]

                # Predict
                prediction = np.argmax(np.dot(self.weights, x_i) + self.biases)

                if prediction != correct_label:
                    # Update correct class vector
                    self.weights[correct_label] += self.learning_rate * x_i
                    self.biases[correct_label] += self.learning_rate
                    # Update wrong class vector
                    self.weights[prediction] -= self.learning_rate * x_i
                    self.biases[prediction] -= self.learning_rate

    def predict(self, X):
        # Prediction result
        return np.argmax(np.dot(X, self.weights.T) + self.biases, axis=1)


# Main
rs_value = 232
learning_rate_value = 0.01
iteration_value = 100

X, y = read_data('Iris.csv')
X, y = shuffle(X, y, random_state=rs_value)

split_index = int(0.7 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Total:", len(y_test))

# model 1
model = DualPerceptronModel(learning_rate=learning_rate_value, iteration=iteration_value)
model.train(X_train, y_train)
preds = model.predict(X_test)

acc = np.mean(preds == y_test)
print("Accuracy:", round(acc * 100, 3), "%  ( Correct:", int(len(y_test)*acc), ")")

# model 2
model2 = MultiClassPerceptron(class_no=3, learning_rate=learning_rate_value, iteration=iteration_value)
model2.train(X_train, y_train)
preds2 = model2.predict(X_test)

acc2 = np.mean(preds2 == y_test)
print("Accuracy:", round(acc2 * 100, 3), "%  ( Correct:", int(len(y_test)*acc2), ")")