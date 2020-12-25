# https://www.kaggle.com/saurabh105/pca-visualization-on-mnist-data-from-scratch
# https://medium.com/analytics-vidhya/principal-component-analysis-pca-with-code-on-mnist-dataset-da7de0d07c22
# https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
# https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

def myPCA(X,k):

    covariance = np.cov(X.T)
    eigen_val, eigen_vector = np.linalg.eigh(covariance)
    eigen_pairs = []
    for i in range(len(eigen_val)):
        eigen_pairs.append(tuple((np.abs(eigen_val[i]), eigen_vector[:, i])))

    eigen_pairs.sort(key =lambda x: x[0], reverse=True)
    pc = []
    for j in range(k):
        pc.append(eigen_pairs[j][1])

    pca = np.asmatrix(pc)
    return pca




data = pd.read_csv("dataset/mnist_train.csv")
# data = pd.read_csv("dataset/iris.csv")
column_names = []
for col in data.columns:
    if col != 'label':
        column_names.append(col)
X = data[column_names]
pca = myPCA(X, 2)

final = X.dot(pca.T)
print("final shape:", final.shape)
expected = data['label'].unique()

final['target'] = data['label']
final = final.rename(columns={0: 'PC1', 1: 'PC2', 'target': 'label'})


sb.scatterplot(final['PC1'], final['PC2'], hue=final['label'], palette ='icefire')
plt.title('Scatter-plot')
plt.show()
plt.title('Q2 part3')

# matrix = np.asmatrix(final.iloc[0, :10])
# print(matrix.reshape(28, 28))



#PART 5
x1 = np.asmatrix(data.iloc[:, 1:].values)
y = data.iloc[:, :1].values
pca = myPCA(X, 30)
final = X.dot(pca.T)
x2 = np.asmatrix(final.iloc[:, :].values)

model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='sigmoid', input_shape=(28, 28, 1)),
        keras.layers.Dense(10, activation='sigmoid')
    ])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam')
start_time1 = time.time()
model.fit(x1, y, epochs=10, verbose=1)
end_time1 = time.time()
print("Training time before PCA: ", end_time1-start_time1)
# model.summary()



model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='sigmoid', input_shape=(32, 31, 1)),
        keras.layers.Dense(10, activation='sigmoid')
    ])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam')


start_time1 = time.time()
model.fit(x2, y, epochs=10, verbose=1)
end_time1 = time.time()
print("Training time after PCA: ", end_time1-start_time1)