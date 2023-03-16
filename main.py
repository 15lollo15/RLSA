import numpy as np
import matplotlib.pyplot as plt

NUM_LAMBDA = 50
MIN_EXP = -15
MAX_EXP = 5
RANDOM_SEED = 99
PROPORTION = 2/3
SIGMA = 6000
SUB_SET_NUM = 10


def load_data():
    x = np.loadtxt('x.csv', delimiter=',')
    y = np.loadtxt('y.csv', delimiter=',')
    return x, y


def gaussian_kernel(ds1: np.matrix, ds2: np.matrix):
    K = np.zeros((ds1.shape[0], ds2.shape[0]))
    print(K.shape)
    for i, x1 in enumerate(ds1):
        for j, x2 in enumerate(ds2):
            K[i, j] = np.exp(-1 * np.linalg.norm(x1 - x2)**2 / (2*SIGMA**2))
    return K


if __name__ == '__main__':
    x, y = load_data()
    n = x.shape[0]
    lambdas = np.logspace(MIN_EXP, MAX_EXP, NUM_LAMBDA)

    np.random.seed(RANDOM_SEED)
    rand_perm = np.random.permutation(n)

    n_train = int(n * PROPORTION)
    n_test = n - n_train

    train = np.take(x, rand_perm[:n_train], axis=0)
    y_train = np.take(y, rand_perm[:n_train], axis=0)

    test = np.take(x, rand_perm[n_train:], axis=0)
    y_test = np.take(y, rand_perm[n_train:], axis=0)

    K_train = gaussian_kernel(train, train)
    K_test = gaussian_kernel(test, train)

    chunk_size = n_train // SUB_SET_NUM

    errors = np.zeros((NUM_LAMBDA, SUB_SET_NUM))
    for i in range(SUB_SET_NUM):
        print("cv with block ", i)
        start = i * chunk_size
        end = (i + 1) * chunk_size

        K_train_cv = np.delete(K_train, np.s_[start:end], axis=0)
        y_train_cv = np.delete(y_train, np.s_[start:end], axis=0)

        K_test_cv = K_train[start:end]
        y_test_cv = y_train[start:end]

        for j, l in enumerate(lambdas):
            a = K_train_cv + l*chunk_size*np.eye(K_train_cv.shape[0], K_train_cv.shape[1])
            c = np.linalg.lstsq(a, y_train_cv, rcond=None)[0]
            prediction = np.sign(np.matmul(K_test_cv, c))
            error = (y_test_cv - prediction)**2
            errors[j, i] = np.mean(error)

    lambda_selected = lambdas[np.argmin(np.mean(errors, axis=1))]
    #lambda_selected = lambdas[25]
    a = K_train + lambda_selected * n_train * np.eye(K_train.shape[0], K_train.shape[1])
    c = np.linalg.lstsq(a, y_train, rcond=None)[0]
    prediction = np.sign(np.matmul(K_test, c))

    error = (y_test - prediction)**2
    wrong_prediction = np.take(test, np.nonzero(error), axis=0)
    print(wrong_prediction.shape)
    fig = plt.figure(figsize=(40, 100))
    print(len(wrong_prediction))
    for i, wp in enumerate(wrong_prediction[0]):
        print(i)
        img = np.reshape(wp, (40, 100), order='F')
        fig.add_subplot(3, 4, i + 1)
        plt.imshow(img, cmap='gray')
    plt.show()











