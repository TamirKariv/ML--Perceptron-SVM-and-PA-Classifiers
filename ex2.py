import numpy as np
import sys


# Perceptron classifier
class Perceptron:
    def __init__(self, X_train, Y_train, X_test, eta=0.01, epochs=38):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.eta = eta
        self.epochs = epochs
        self.weights = np.zeros((3, X_train.shape[1]))

    # train the model
    def train_model(self):
        for e in range(self.epochs):
            for x, y in zip(self.X_train, self.Y_train):
                y = int(y)
                y_hat = np.argmax(np.dot(self.weights, x))
                y_hat = int(y_hat)
                if y != y_hat:
                    self.weights[y, :] = self.weights[y, :] + self.eta * x
                    self.weights[y_hat, :] = self.weights[y_hat, :] - self.eta * x


# Passive Aggressive classifier
class Passive_Aggressive:
    def __init__(self, X_train, Y_train, X_test, epochs=70):
        self.Y_train = Y_train
        self.X_test = X_test
        self.epochs = epochs
        self.X_train = X_train
        self.weights = np.zeros((3, X_train.shape[1]))

    # train the model
    def train_model(self):
        for e in range(self.epochs):
            for x, y in zip(self.X_train, self.Y_train):
                y = int(y)
                y_hat = np.argmax(np.dot(self.weights, x))
                y_hat = int(y_hat)
                if y != y_hat:
                    hinge_loss = max(0, 1 - np.dot(self.weights[y, :], x) + np.dot(self.weights[y_hat, :], x))
                    n = (np.linalg.norm(x) ** 2) * 2
                    if n == 0:
                        break
                    self.weights[y, :] = self.weights[y, :] + ((hinge_loss / n) * x)
                    self.weights[y_hat, :] = self.weights[y_hat, :] - ((hinge_loss / n) * x)


# SVM classifier
class SVM:
    def __init__(self, X_train, Y_train, X_test, eta=0.01, epochs=90, Lambda=0.01):
        self.epochs = epochs
        self.eta = eta
        self.Lambda = Lambda
        self.Beta = 1 - self.eta * self.Lambda
        self.Y_train = Y_train
        self.X_train = X_train
        self.X_test = X_test
        self.weights = np.zeros((3, X_train.shape[1]))

    # train the model
    def train_model(self):
        for e in range(self.epochs):
            for x, y in zip(self.X_train, self.Y_train):
                y = int(y)
                y_hat = np.argmax(np.dot(self.weights, x))
                y_hat = int(y_hat)
                idx = list({0, 1, 2} - {y, y_hat})[0]
                self.weights[idx, :] = self.weights[idx, :] * self.Beta
                if y != y_hat:
                    self.weights[y, :] = self.weights[y, :] * self.Beta + self.eta * x
                    self.weights[y_hat, :] = self.weights[y_hat, :] * self.Beta - self.eta * x


# test the model return the results
def test_model(m_weights, test_x):
    results = np.zeros(test_x.shape[0])
    for idx, x in enumerate(test_x):
        y_hat = np.argmax(np.dot(m_weights, x))
        results[idx] = y_hat

    return results


# get the inputs from the file add non-numerical features using OneHotEncoder
def get_inputs(filename):
    numerical_fields = np.loadtxt(filename, dtype=float, delimiter=",", usecols=(1, 2, 3, 4, 5, 6, 7))
    non_numerical_fields = np.loadtxt(filename, dtype=str, delimiter=",", usecols=0)
    added_fields = np.zeros((non_numerical_fields.size, 3))
    for i in range(added_fields.shape[0]):
        if non_numerical_fields[i] == 'M':
            added_fields[i][0] = 1
        elif non_numerical_fields[i] == 'F':
            added_fields[i][1] = 1
        else:
            added_fields[i][2] = 1
    data = np.hstack([added_fields, numerical_fields])
    return data


# normalize the inputs using min-max
def normalize_data(data):
    normalized_data = np.zeros(0)
    for i in range(data.shape[1]):
        col = data[:, i]
        min_elm = np.amin(col)
        max_elm = np.amax(col)
        if min_elm == max_elm:
            col = col * 0
        else:
            col = (col - min_elm) / (max_elm - min_elm)
        if i == 0:
            normalized_data = col
        else:
            normalized_data = np.column_stack(([normalized_data, col]))
    return normalized_data


def main():
    # get the data and normalize the inputs
    train_X = normalize_data(get_inputs(sys.argv[1]))
    train_Y = np.loadtxt(sys.argv[2], dtype=float).reshape(-1, 1)
    test_X = normalize_data(get_inputs(sys.argv[3]))

    # train the models
    perceptron = Perceptron(train_X, train_Y, test_X)
    svm = SVM(train_X, train_Y, test_X)
    passive_aggresive = Passive_Aggressive(train_X, train_Y, test_X)
    perceptron.train_model()
    passive_aggresive.train_model()
    svm.train_model()

    # get the results from the test set
    perceptron_results = test_model(perceptron.weights, perceptron.X_test)
    svm_results = test_model(svm.weights, svm.X_test)
    passive_aggresive_results = test_model(passive_aggresive.weights, passive_aggresive.X_test)

    # print the results
    for i in range(perceptron_results.size):
        print('perceptron: {}, svm: {}, pa: {}'.format(str(int(perceptron_results[i])), str(int(svm_results[i])),
                                                       str(int(passive_aggresive_results[i]))))


if __name__ == "__main__":
    main()
