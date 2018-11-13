from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math


def load_data():
    return loadmat('logreg.mat')


def f(w0, w, x, y):
    n = len(y)

    return np.mean(np.log(np.add(np.full(n, 1), np.exp(np.add(np.full(n, w0), np.dot(x, w))))).reshape(n, 1) -
                   y * (np.add(np.full(n, w0), np.dot(x, w))).reshape(n, 1))


def grad(w0, w, x, y):
    n = len(y)

    return np.mean(np.subtract(
        np.divide(np.multiply(np.exp(np.add(np.full(n, w0), np.dot(x, w))).reshape(n, 1), x),
                  np.add(np.full(n, 1), np.exp(np.add(np.full(n, w0), np.dot(x, w)))).reshape(n, 1)),
        np.multiply(y, x)), axis=0)


def grad_0(w0, w, x, y):
    n = len(y)

    # return np.add(np.full(n, 1), np.exp(np.add(np.full(n, 0), np.dot(x, w))))
    return np.mean(np.subtract(
        np.divide(np.exp(np.add(np.full(n, w0), np.dot(x, w))).reshape(n, 1),
                  np.add(np.full(n, 1), np.exp(np.add(np.full(n, w0), np.dot(x, w)))).reshape(n, 1)), y))


def backtracking(w0, w, x, y):
    d = len(x[0])
    n = len(y)

    eta = 1
    # lambda1 = (1/n)*np.sum(np.subtract(1/2, y))
    # lambda2 = np.divide(np.sum(np.subtract(np.divide(x, 2), np.multiply(y, x)), 0), n)
    lambda1 = grad_0(w0, w, x, y)
    lambda2 = grad(w0, w, x, y)
    # lambda2 = np.multiply(y, x)

    # print(lambda1)
    # print(lambda1)
    # print(lambda2)

    while True:
        w0_minus_etalambda = w0 - eta * lambda1
        # w_minus_etalambda = [-1 * eta * lambda1, -1*eta*lambda2]
        w_minus_etalambda = np.subtract(w, eta * lambda2)
        # LHS = f(0-eta*lambda1, -1*eta*lambda2, x, y)
        LHS = f(w0_minus_etalambda, w_minus_etalambda, x, y)
        RHS = f(w0, w, x, y) - (1 / 2) * eta * (np.linalg.norm(lambda1) ** 2 + np.linalg.norm(lambda2) ** 2)

        if LHS > RHS:
            eta = eta / 2
        else:
            break
    # print(eta)
    return eta


def grad_descent(data, labels):
    n = len(labels)
    d = len(data[0])

    beta0 = 0
    beta = np.zeros(d)

    iterations = 0
    while True:
        iterations += 1
        if iterations % 100 == 0:
            print(iterations)

        eta = backtracking(beta0, beta, data, labels)

        l = grad(beta0, beta, data, labels)
        l0 = grad_0(beta0, beta, data, labels)
        # print(l)
        # print(l0)

        beta = beta - eta * l
        beta0 = beta0 - eta * l0

        func = f(beta0, beta, data, labels)
        #print(func)
        if func <= 0.65064:
            break

    return iterations


def mod_grad_descent(training, test, training_labels, test_labels):
    data = training
    labels = training_labels

    n = len(training)
    d = len(data[0])

    beta0 = 0
    beta = np.zeros(d)

    iterations = 0
    best = 1
    while True:
        iterations += 1
        if iterations % 100 == 0:
            print(iterations)
        if is_pow2(iterations):
            validation_er = zero_one_loss(beta0, beta, test, test_labels)
            if validation_er > 0.99 * best and iterations >= 32:
                break
            else:
                best = min(best, validation_er)


        eta = backtracking(beta0, beta, data, labels)

        l = grad(beta0, beta, data, labels)
        l0 = grad_0(beta0, beta, data, labels)
        # print(l)
        # print(l0)

        beta = beta - eta * l
        beta0 = beta0 - eta * l0

        func = f(beta0, beta, data, labels)
        # print(func)
    print('Validation error: {}'.format(validation_er))

    func = f(beta0, beta, data, labels)
    print('Objective function value: {}'.format(func))

    return iterations


def is_pow2(n):
    return n!=0 and ((n & (n - 1)) == 0)


def zero_one_loss(beta0, beta, test, test_labels):
    n = len(test_labels)
    error = 0
    for i in range(n):
        temp_label = test_labels[i]
        if temp_label == 0:
            temp_label = -1
        if (beta0 + np.dot(beta, test[i])) * temp_label <= 0:
            error += 1
    return error/n


if __name__ == '__main__':
    logreg = load_data()

    data = logreg['data']
    labels = logreg['labels']


    n = len(labels)

    # iterations = grad_descent(data, labels)
    # print(iterations)
    #
    # #linear transformation, scale first and last parameters by 0.1
    # mod_data = np.multiply(data, [.1, 1, .1])
    #
    # iterations = grad_descent(mod_data, labels)
    # print(iterations)

    print('modified gradient descent, original data:')
    training = data[:math.floor(0.8*n)]
    test = data[math.floor(0.8*n):]
    training_labels = labels[:math.floor(0.8*n)]
    test_labels = labels[math.floor(0.8*n):]

    # print(len(training))
    # print(len(test))
    # print(n)

    iterations = mod_grad_descent(training, test, training_labels, test_labels)
    print(iterations)

    print('modified gradient descent, transformed data:')
    data = np.multiply(data, [.1, 1, .1])
    training = data[:math.floor(0.8*n)]
    test = data[math.floor(0.8*n):]
    training_labels = labels[:math.floor(0.8*n)]
    test_labels = labels[math.floor(0.8*n):]

    iterations = mod_grad_descent(training, test, training_labels, test_labels)
    print(iterations)
