import numpy as np
import matplotlib.pyplot as plt
import math
import random

def dataset(mu_values, sigma, N, k):
    X_initial = []
    for idx, i in enumerate(mu_values):
        X_initial.append(np.random.multivariate_normal(i, sigma, N))

    X = np.concatenate((X_initial[0], X_initial[1], X_initial[2]))
    dist = 0
    avg = []
    if k == 3:
        for i in range(0,3):

            for item in range(len(X_initial[i])):
                dist += euclidean_distance(X_initial[i][item], mu_values[i])
            avg.append(dist/len(X_initial[i]))
    return X, avg

def euclidean_distance(x1, x2):
    squared_distance = 0

    for i in range(len(x1)):
        squared_distance += (x1[i] - x2[i]) ** 2

    ed = math.sqrt(squared_distance)

    return ed;

def mykmeans(X,k, tol = 0.0001, max_iterations=10000):

    clusters = {}
    centroid = []
    for i in range(0,k):
        centroid.append(random.choice(X))
    for x in range(max_iterations):
        for i in centroid:
            clusters[tuple(i)] = []

        for data in X:
            index = 0
            distance = []
            average = 0
            for c in centroid:
                distance.append(euclidean_distance(c, data))
            min_dist = min(distance)
            index = distance.index(min_dist)
            prev_centroid = centroid[index]
            clusters[tuple(centroid[index])].append(data)
            average = np.average(clusters[tuple(centroid[index])], axis=0)
            if np.sum((average - prev_centroid) / prev_centroid * 100) > tol:
                centroid[index] = average
                clusters[tuple(average)] = clusters.pop(tuple(prev_centroid))
            else:
                centroid[index] = centroid[index]

        return clusters, centroid


if __name__ == "__main__":

    # PARAMETERS
    mu_values = [[[-3, 0], [3, 0], [0, 3]], [[-2, 0], [2, 0], [0, 2]]]
    sigma = np.array([[1,0.75],[0.75,1]])
    k_val = [2,3,4,5]
    N = 300

    for i in k_val:
        for mu in mu_values:
            n = 0
            X, avg = dataset(mu, sigma, N, i)
            plt.scatter(X[n:n + N][:, 0], X[n:n + N][:, 1])
            n = n + N
            plt.scatter(X[n:n + N][:, 0], X[n:n + N][:, 1])
            n = n + N
            plt.scatter(X[n:n + N][:, 0], X[n:n + N][:, 1])
            # plt.show()
            # plt.savefig('Inital_Data_{}'.format(mu))
            clusters, centroid = mykmeans(X, i)

            if i == 3:
                print("old averages:", avg, "for mu:", mu)
                new_avg = []
                for dict_key in clusters:
                    cluster = clusters[dict_key]
                    new_dist = 0
                    for points in cluster:
                        new_dist = euclidean_distance(points, dict_key)
                    new_avg.append(new_dist/len(cluster))
                print("new averages:", new_avg, "for new centroids:", clusters.keys())

            xplx = []
            yplx = []
            cluster = []
            for dict_key in clusters:
                cluster = clusters[dict_key]

                xplx, yplx = map(list, zip(*cluster))
                plt.scatter(xplx, yplx)
                plt.scatter(cluster[0][0], cluster[0][1], marker='X', s=20 * 4)
            plt.show()
            plt.savefig('for_{}_mu_{}.png'.format(i, mu), format='png')