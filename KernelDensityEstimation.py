import numpy as np
import matplotlib.pyplot as plt


def mykde(X, h):
    X_range = np.array([np.amin(X), np.amax(X)])
    partition_size = .05
    binLocation = X_range[0] + partition_size
    no_bins = int(X_range[1] / partition_size)
    probabilities = np.zeros((no_bins, 2))
    Counter = 0
    while (Counter < no_bins):
        pointsInBin = 0
        for i in X:
            dist_bin = ((binLocation - i) / h)
            if (abs(dist_bin) <= .5):
                pointsInBin += 1

        probabilities[Counter] = ([(1 / (len(X) * h)) * pointsInBin, binLocation])

        Counter += 1
        binLocation += partition_size
    probs = np.array(probabilities)
    return probs, X_range


def mykde2D(X, h):
    x_range = np.array(([np.amin(X[:, 0]), np.amax(X[:, 0])]))
    y_range = np.array(([np.amin(X[:, 1]), np.amax(X[:, 1])]))
    xPartitionSize = abs(x_range[0]) + abs(x_range[1])
    yPartitionSize = abs(y_range[0]) + abs(y_range[1])

    no_xbins = 10
    no_ybins = 10
    xPartitionSize /= no_xbins
    yPartitionSize /= no_ybins
    probabilities = np.zeros(((no_xbins + 1) * no_ybins, 3))
    loop_counter = 0
    xPoint = x_range[0]
    while (loop_counter < (len(probabilities))):
        yPoint = y_range[0]
        for j in range(no_ybins):
            pointsInBin = 0
            for k in X:
                point = np.array((xPoint, yPoint))
                distanceToBin = ((np.linalg.norm(point - k)) / h)
                if (abs(distanceToBin) <= .5):
                    pointsInBin += 1
            yPoint += yPartitionSize
            probabilities[loop_counter, 0] = xPoint
            probabilities[loop_counter, 1] = yPoint
            probabilities[loop_counter, 2] = ((1 / (len(X) * h )) * pointsInBin)
            loop_counter += 1

        xPoint += xPartitionSize

    probs = np.array(probabilities)
    domain = np.concatenate((x_range, y_range), axis=0)
    return probs, domain




data1 = np.random.normal(5, 1, 1000)
h = np.array([.1, 1, 5, 10])


fig, axs = plt.subplots(2, 2)

subTitle = 'h =' + str(h[0])
p, x = mykde(data1, h[0])
axs[0, 0].scatter(p[:, 1], p[:, 0], s=2, c='b')
axs[0, 0].hist(data1, 20, alpha=.5, density=True, color='pink')
axs[0, 0].set_title(subTitle)

subTitle = 'h =' + str(h[1])
p, x = mykde(data1, h[1])
axs[0, 1].scatter(p[:, 1], p[:, 0], s=2, c='b')
axs[0, 1].hist(data1, 20, alpha=.5, density=True, color='pink')
axs[0, 1].set_title(subTitle)

subTitle = 'h =' + str(h[2])
p, x = mykde(data1, h[2])
axs[1, 0].scatter(p[:, 1], p[:, 0], s=2, c='b')
axs[1, 0].hist(data1, 20, alpha=.5, density=True, color='pink')
axs[1, 0].set_title(subTitle)

subTitle = 'h =' + str(h[3])
p, x = mykde(data1, h[3])
axs[1, 1].scatter(p[:, 1], p[:, 0], s=2, c='b')
axs[1, 1].hist(data1, 20, alpha=.5, density=True, color='pink')
axs[1, 1].set_title(subTitle)

plt.suptitle('Part 1 of 1')
plt.show()


data2 = np.random.normal(0, 0.2, 1000)
combined = np.concatenate((data1, data2))


fig, ax = plt.subplots(2, 2)

subTitle = 'h =' + str(h[0])
p, x = mykde(combined, h[0])
ax[0, 0].scatter(p[:, 1], p[:, 0], s=2, c='b')
ax[0, 0].hist(combined, 20, alpha=.5, color='pink', density=True)
ax[0, 0].set_title(subTitle)

subTitle = 'h =' + str(h[1])
p, x = mykde(combined, h[1])
ax[0, 1].scatter(p[:, 1], p[:, 0], s=2, c='b')
ax[0, 1].hist(combined, 20, alpha=.5, color='pink', density=True)
ax[0, 1].set_title(subTitle)

subTitle = 'h =' + str(h[2])
p, x = mykde(combined, h[2])
ax[1, 0].scatter(p[:, 1], p[:, 0], s=2, c='b')
ax[1, 0].hist(combined, 20, alpha=.5, color='pink', density=True)
ax[1, 0].set_title(subTitle)

subTitle = 'h =' + str(h[3])
p, x = mykde(combined, h[3])
ax[1, 1].scatter(p[:, 1], p[:, 0], s=2, c='b')
ax[1, 1].hist(combined, 20, alpha=.5, color='pink', density=True)
ax[1, 1].set_title(subTitle)

plt.suptitle('Part 2 of 1')
plt.show()

mu1 = np.array([1, 0])
dev1 = np.array([[0.9, 0.4], [0.4, 0.9]])
mu2 = np.array([0, 2.5])
dev2 = ([[0.9, 0.4], [0.4, 0.9]])

data1_2d = np.random.multivariate_normal(mu1, dev1, 500)
data2_2d = np.random.multivariate_normal(mu2, dev2, 500)
X2 = np.concatenate((data1_2d, data2_2d), axis=0)

fig2, ax2 = plt.subplots(2, 2)

subTitle = 'h =' + str(h[0])
p, x = mykde2D(X2, h[0])
size = p[:, 2]
size *= 1000
ax2[0, 0].scatter(X2[:, 0], X2[:, 1])
ax2[0, 0].scatter(p[:, 0], p[:, 1], c='pink', alpha=.9, s=size)
ax2[0, 0].set_title(subTitle)

subTitle = 'h =' + str(h[1])
p, x = mykde2D(X2, h[1])
size = p[:, 2]
size *= 1000
ax2[0, 1].scatter(X2[:, 0], X2[:, 1])
ax2[0, 1].scatter(p[:, 0], p[:, 1], c='pink', alpha=.9, s=size)
ax2[0, 1].set_title(subTitle)

subTitle = 'h =' + str(h[2])
p, x = mykde2D(X2, h[2])
size = p[:, 2]
size *= 1000
ax2[1, 0].scatter(X2[:, 0], X2[:, 1])
ax2[1, 0].scatter(p[:, 0], p[:, 1], c='pink', alpha=.9, s=size)
ax2[1, 0].set_title(subTitle)

subTitle = 'h =' + str(h[3])
p, x = mykde2D(X2, h[3])
size = p[:, 2]
size *= 1000
ax2[1, 1].scatter(X2[:, 0], X2[:, 1])
ax2[1, 1].scatter(p[:, 0], p[:, 1], c='pink', alpha=.9, s=size)
ax2[1, 1].set_title(subTitle)

plt.suptitle('2D Data')
plt.tight_layout()
plt.show()