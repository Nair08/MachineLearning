# https://www.geeksforgeeks.org/python-mean-squared-error/
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    squared_distance = 0
    for i in range(len(x1)):
        squared_distance += (x1[i] - x2[i]) ** 2

    ed = math.sqrt(squared_distance)

    return ed

def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    top_dist = []
    for i in range(k):
        neighbors.append(distances[i][0])
        top_dist.append(distances[i][1])

    return neighbors, top_dist

def myknnclassify(X, test, k):
    neighbors, top_dist = get_neighbors(X, test, k)
    # print("neighbour:", neighbors)
    output = [row[-1] for row in neighbors]
    prediction = max(set(output), key=output.count)
    return neighbors, prediction

def myknnregress(X, test, k):
    neighbors, top_dist = get_neighbors(X, test, k)
    avg = np.mean([row[-1] for row in neighbors])
    return avg


def myLWR(X, test, k):
    neighbors, top_dist = get_neighbors(X, test, k)
    weight = 0
    # print("neighbour: ", neighbors)
    for i in range(k):
        w = 1/top_dist[i]
        weight = (w*neighbors[i][-1]) + weight
    return weight




if __name__ == "__main__":

    # PARAMETERS
    # np.random.seed(2)
    mu0 = [1, 0]
    mu1 = [0, 1]
    sigma0 = np.array([[1, 0.75], [0.75, 1]])
    sigma1 = np.array([[1, -0.5], [0.5, 1]])
    k_val = [1,2,3,4,5,10,20]
    data0 = np.random.multivariate_normal(mu0, sigma0, 200)
    data1 = np.random.multivariate_normal(mu1, sigma1, 200)
    train0 = pd.DataFrame({'Xpoint':data0[:, 0], 'Ypoints':data0[:, 1], 'Class:':0})
    train1 = pd.DataFrame({'Xpoint': data1[:, 0], 'Ypoints': data1[:, 1], 'Class:': 1})
    train_data = pd.concat([train0, train1])
    train_set = train_data.to_numpy()
    # print(train_set)
    tdata0 = np.random.multivariate_normal(mu0, sigma0, 50)
    tdata1 = np.random.multivariate_normal(mu1, sigma1, 50)
    test0 = pd.DataFrame({'Xpoint1': tdata0[:, 0], 'Ypoints': tdata0[:, 1], 'Class:': 0})
    test1 = pd.DataFrame({'Xpoint1': tdata1[:, 0], 'Ypoints': tdata1[:, 1], 'Class:': 1})
    test_data = pd.concat([test0, test1])
    test_set = test_data.to_numpy()
    for idx,i in enumerate(train_set):
        # print(train_set[idx][2])
        plt.scatter(train_set[idx][0],train_set[idx][1],c=train_set[idx][2])
    plt.show()
    # print(test_set)
    # neighbour = []
    # for i in train_set:
    #     neighbour.append(get_neighbors(train_set,i,3))
    # print(neighbour)
    actual = []
    predictions =[]
    for k in k_val:
        errors = []
        for i in test_set:
            neighbors, predict = myknnclassify(train_set, i, k)
            predictions.append(predict)
            actual.append(i[-1])
        accuracy = 0
        count = 0
        # for i in range(len(actual)):
        #
        #     if actual[i] == predictions[i]:
        #         count = count +1
        # accuracy = (count/len(predictions))*100
        # print(accuracy)
        MSE1 = np.square(np.subtract(actual, predictions)).mean()
        errors.append(MSE1)
        print("error for 1", errors)


###-----P2.2-----

    data2 = np.random.multivariate_normal(mu0, sigma0, 300)
    x1 = data2[:, 0]
    x2 = data2[:, 1]
    e = np.random.normal(0,0.5)
    y = 2*x1 + x2 + e
    train2 = pd.DataFrame({'Xpoint': x1, 'Ypoints': x2, 'Class:': y})
    train2_set = train2.to_numpy()

    tdata2 = np.random.multivariate_normal(mu0, sigma0, 300)
    tx1 = tdata2[:, 0]
    tx2 = tdata2[:, 1]
    e = np.random.normal(0, 0.5)
    ty = 2*x1 + x2 + e
    test2 = pd.DataFrame({'Xpoint': tx1, 'Ypoints': tx2, 'Class:': ty})
    test2_set = test2.to_numpy()
    actual2 = []
    predictions2 = []
    predictions3 = []
    for k in k_val:
        errors2 = []
        errors3 = []
        for i in test2_set:
            average = myknnregress(train2_set,i,k)
            # print("new class:", average)
            predictions2.append(average)
            actual2.append(i[-1])

            weights = myLWR(train2_set,i,k)
            # print("weight", weights)
            predictions3.append(weights)
        MSE2 = np.square(np.subtract(actual2, predictions2)).mean()
        errors2.append(MSE2)
        print("Error for 2", errors2)
        # MSE3 = np.square(np.subtract(actual2, predictions3)).mean()
        # errors3.append(MSE3)
        # print("Error for 3", errors3)



