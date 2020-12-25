import numpy as np
import idx2numpy
import math


def cross_entropy(a, y):
    return -sum([y[j] * math.log(a[j]) for j in range(len(y))])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def flatten(train):
    X_train = train / 255.0
    X_train = np.array([X.flatten() for X in X_train])
    return X_train

class multiclass:
    def __init__(self, lr=0.1, batch_size=200, tol=0.1):

        self.lr = lr
        self.batch_size = batch_size
        self.tolerance = tol
        self.iterations = 0
        self.actual = None

    def train(self, train, label, classes=5):

        batch_size = self.batch_size
        train_size = train.shape[0]
        train_idx = np.arange(train_size)
        batch = np.array_split(train_idx, train_size // batch_size + 1)
        W = np.random.random((classes, 784)) * 0.01
        bias = 1
        loss_list = []

        for i in range(10000):
            for b in batch:
                X = train[b]
                y = label[b]

                z = np.dot(X, W.T) + bias  # Shape = (32,5)

                a = np.zeros((len(b), classes))

                for idx, val in enumerate(z):
                    a[idx] = softmax(val)

                dz = a - y  # Shape = (32,5)
                dw = np.dot(dz.T, X) / len(b)  # Shape = (5,764)
                db = np.sum(dz) / len(b)

            loss = cross_entropy(a[-1].reshape(classes, 1), y[-1].reshape(classes, 1))[0]

            loss_list.append(loss)

            if loss < self.tolerance:
                print("loss less than tolerance at iteration =  ", i)
                break

            if i % 100 == 0:
                print("Loss:", loss, "iteration:", i)

            W = W - self.lr * dw
            bias = bias - self.lr * db

        self.W = W
        self.bias = bias
        self.losslist = loss_list

    def predict(self, test, l):

        actual = l.reshape((test.shape[0], 1))
        W = self.W
        bias = self.bias

        X = np.array([X.flatten() for X in test])

        z = np.dot(X, W.T) + bias
        predictions = np.zeros((test.shape[0], 1))
        acc = 0

        for i, val in enumerate(z):
            prediction = np.argmax(softmax(val))
            predictions[i] = prediction
            if prediction == actual[i]:
                acc += 1

        accuracy = (acc / test.shape[0]) * 100
        print('Accuracy =', accuracy)

        return predictions, accuracy

    def precision_Recall(self, actual, predicted):

        PrecisionList = []
        RecallList = []

        confusionMatrix = np.zeros((5, 5))

        for i in range(len(predicted)):
            pred = int(predicted[i])
            a = actual[i]

            if pred == a:
                confusionMatrix[pred][pred] += 1
            else:
                confusionMatrix[pred][a] += 1

        for c in np.unique(actual):
            TP = confusionMatrix[c][c]
            TN = 0
            FP = 0
            FN = 0
            for row in np.unique(actual):
                for column in np.unique(actual):
                    if column != c and row != c:
                        TN += confusionMatrix[row][column]
                    if row == c and column != c:
                        FP += confusionMatrix[row][column]
                    if row != c and column == c:
                        FN += confusionMatrix[row][column]

            RecallList.append(TP / (TP + FN))
            PrecisionList.append(TP / (TP + FP))

        return confusionMatrix, PrecisionList, RecallList


if __name__ == "__main__":
    batch_size = 200
    tolerance = 0.01
    learningRate = 0.01

    # X_train, y_train = idx2numpy.convert_from_file('D:/UTA/Fall_2020/MachineLearning/Assignments/Assignment2/train-images.idx3-ubyte'), idx2numpy.convert_from_file('D:/UTA/Fall_2020/MachineLearning/Assignments/Assignment2/train-labels.idx1-ubyte')
    X_train, y_train = idx2numpy.convert_from_file('train-images.idx3-ubyte'),idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    # X_test, y_test = idx2numpy.convert_from_file('D:/UTA/Fall_2020/MachineLearning/Assignments/Assignment2/t10k-images.idx3-ubyte'), idx2numpy.convert_from_file('D:/UTA/Fall_2020/MachineLearning/Assignments/Assignment2/t10k-labels.idx1-ubyte')
    X_test, y_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte'), idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    train_index = np.where(y_train < 5)[0]
    test_index = np.where(y_test < 5)[0]

    X_train, y_train = X_train[train_index], y_train[train_index]
    X_test, y_test = X_test[test_index], y_test[test_index]

    X_test = X_test / 255.0
    X_train = flatten(X_train)
    class_no = len(np.unique(y_train))
    encoder = np.zeros((len(y_train), class_no))
    for i in range(len(encoder)):
        encoder[i][y_train[i]] = 1

    model = multiclass(learningRate, batch_size, tolerance)
    model.train(X_train, encoder)

    op = model.predict(X_test, y_test)
    conf_matrix, precision, recall = model.precision_Recall(y_test, op)

    for idx, class_no in enumerate(np.unique(y_test)):
        print("Class:", class_no, "precison is:", precision[idx], "Recall value:", recall[idx])
