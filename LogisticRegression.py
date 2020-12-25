import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy(a, data):
    m = np.shape(data)[0]
    Y = data[:, 2:]
    A = a
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(data[:, :2].T, (A - Y))
    db = 1 / m * np.sum(A - Y)

    return cost, dw, db


def featureDistance(w):
    sum = 0
    for weight in w:
        sum += weight * weight

    return math.sqrt(sum)


def logistic_reg(train, batch_size, lr, num_epochs=10000, tolerance=0.0001):
    oldcost = 1
    iterations = 0
    for j in range(num_epochs):
        for s in range(0, np.shape(train)[0], batch_size):
            batch = train[s: s + batch_size]
            if 0 <= s <= batch_size:
                w0 = np.zeros((2, 1))
                b0 = 0
            else:
                w0 = w
                b0 = b
            z0 = np.dot(batch[:, :2], w0) + b0
            a0 = sigmoid(z0)
            cost, dw, db = cross_entropy(a0, batch)
            w = w0 - lr * dw
            b = b0 - lr * db
        if lr * featureDistance(w[0]) < tolerance and cost - oldcost < tolerance:

            print("break")
            break
        oldcost = cost
    iterations = j
    return w, b, iterations


def confusionMatrix(actual, predictions):

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for p in range(0, len(actual)):
        if actual[p] == predictions[p] == 1:
            TP += 1
        if actual[p] == predictions[p] == 0:
            TN += 1
        if predictions[p] == 1 and actual[p] != predictions[p]:
            FP += 1
        if predictions[p] == 0 and actual[p] != predictions[p]:
            FN += 1
    return TP, FP, TN, FN


def area_under_curve(x, y):
    area = 0
    for a in range(len(x)):
        if i != 0:
            area += (x[a] - x[a - 1]) * y[a - 1] + (0.5 * (x[a] - x[a - 1]) * (y[a] - y[a - 1]))
    return area


def prediction(test, weights, bias):
    pred = []
    z = np.dot(test[:, :2], weights) + bias
    a = sigmoid(z)
    class_label = test[:, 3:]
    for i in range(1, np.shape(a)[0]):
        pred = np.where(a > 0.5, 1, 0)

    count = 0
    labels = pd.DataFrame({'Actual': test[:, 2], 'Predictions': pred[:, 0]})

    labels.Actual = labels.Actual.apply(lambda x: int(x))
    for l in range(0, len(labels)):
        if labels.Actual[l] == labels.Predictions[l]:
            count += 1
    accuracy = count / len(labels) * 100

    tpr = []
    fpr = []
    roc_pred = []
    for j in range(0, 1000, 10):
        threshold = j/100
        roc_pred = np.where(a > threshold, 1, 0)

        TP, FP, TN, FN = confusionMatrix(test[:, 2], roc_pred)
        tpr.append(TP / 500)
        fpr.append(FP / 500)
    tpr.append(0)
    fpr.append(0)
    auc = area_under_curve(tpr, fpr)
    plt.plot(fpr, tpr)
    plt.title(f'ROC , Area Under Curve = {auc}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return accuracy


if __name__ == "__main__":
    # PARAMETERS
    mu0 = [1, 0]
    mu1 = [0, 1]
    sigma0 = np.array([[1, 0.75], [0.75, 1]])
    sigma1 = np.array([[1, -0.5], [0.5, 1]])
    data0 = np.random.multivariate_normal(mu0, sigma0, 1000)
    data1 = np.random.multivariate_normal(mu1, sigma1, 1000)
    train0 = pd.DataFrame({'Xpoint': data0[:, 0], 'Ypoints': data0[:, 1], 'Class:': 0})
    train1 = pd.DataFrame({'Xpoint': data1[:, 0], 'Ypoints': data1[:, 1], 'Class:': 1})
    train_data = pd.concat([train0, train1])
    train_set = train_data.to_numpy()

    tdata0 = np.random.multivariate_normal(mu0, sigma0, 500)
    tdata1 = np.random.multivariate_normal(mu1, sigma1, 500)
    test0 = pd.DataFrame({'Xpoint1': tdata0[:, 0], 'Ypoints': tdata0[:, 1], 'Class:': 0})
    test1 = pd.DataFrame({'Xpoint1': tdata1[:, 0], 'Ypoints': tdata1[:, 1], 'Class:': 1})
    test_data = pd.concat([test0, test1])
    test_set = test_data.to_numpy()

    # 0.0001,0.001, 0.01, 0.1, and 1
    learing_rates = [0.0001, 0.001, 0.01, 0.1, 1]
    for i in learing_rates:
        weights, bias, iterations = logistic_reg(train_set, 32, i)
        print("learning rate", i, "No of iterations:", iterations)
        op = prediction(test_set, weights, bias)
        print("accuracy", op)

""" For each learning rate we have to
    1. Create batches
    2. Calculate weights and biases
    3. Repeat for every batch
    4. Predict after all the batches are done  
"""
