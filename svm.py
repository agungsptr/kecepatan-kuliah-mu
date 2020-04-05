import numpy as np


def find_hyperplane():
    # data_set
    X = np.array([
        [3.06, 3.16, 3.17, 3.17, 3.17, -1],
        [2.43, 2.61, 2.49, 2.64, 2.67, -1],
        [3.41, 3.43, 3.43, 3.44, 3.44, -1],
        [3.50, 3.53, 3.54, 3.53, 3.53, -1],
        [2.07, 2.22, 2.28, 2.40, 2.32, -1],
        [3.63, 3.60, 3.58, 3.60, 3.61, -1],
        [2.10, 2.36, 2.35, 2.51, 2.55, -1],
        [2.80, 2.94, 2.94, 3.06, 3.06, -1],
        [3.13, 3.11, 3.10, 3.10, 3.10, -1],
        [2.48, 2.60, 2.61, 2.74, 2.72, -1]
    ])

    # inisiasi label
    Y = np.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1])

    # inisiasi bobot untuk hyperplane
    w = np.zeros(len(X[0]))

    # tingkat pembelajaran
    eta = 1

    # jumlah iterasi
    epochs = 25000

    # jumlah error
    errors = []

    # training untuk menentukan gradient hyperplane
    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            # misclassification
            if (Y[i] * np.dot(X[i], w)) < 1:
                # misclassified update for ours weights
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
                error = 1
            else:
                # correct classification, update our weights
                w = w + eta * (-2 * (1 / epoch) * w)
        errors.append(error)

    return w
