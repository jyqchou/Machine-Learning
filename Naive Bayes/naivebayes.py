#!/usr/bin/env python

from __future__ import print_function
from scipy.io import loadmat
import numpy as np
from pprint import pprint

def estimate_naive_bayes_classifier(X,Y):
    X = X.toarray()

    n = X.shape[0]
    d = X.shape[1]
    k = np.max(Y) - np.min(Y) + 1
    base = np.min(Y)

    print('n = %d' % n)
    print('d = %d' % d)
    print('k = %d' % k)

    size_per_y = np.zeros(shape=k)
    param_pi = np.zeros(shape=k)

    for i in range(n):
        label = Y[i] - base
        size_per_y[label] += 1

    param_pi = np.divide(size_per_y, n)

    #print(size_per_y)
    #print(param_pi)

    param_mu = np.zeros(shape=(k,d))

    for i in range(n):
        #if i%100 == 0:
            #print(i)
        features = X[i]
        label_index = Y[i]-base
        param_mu[label_index] = np.add(param_mu[label_index], features)

    #print(param_mu)

    for i in range(k):
        param_mu[i] = np.divide(np.add(1,param_mu[i]), np.add(2, size_per_y[i]))
    print(param_mu)

    params = [param_pi, param_mu]
    #print(params)
    #print(params[0].shape)
    #print(params[1].shape)

    return params

def predict(params,X):
    print('predicting:')
    param_pi = params[0]
    param_mu = params[1]

    n=X.shape[0]
    d=X.shape[1]
    k=param_pi.shape[0]
    print('n = %d, d = %d, k = %d' % (n, d, k))

    X = X.toarray()

    term1 = np.log(np.array(param_pi)*np.prod(np.subtract(1, param_mu), axis=1))
    #print(term1)
    #print(term1.shape)

    term2 = np.matmul(X, np.transpose(np.log(np.divide(param_mu, np.subtract(1, param_mu)))))
    #print(term2)
    #print(term2.shape)

    sum = term2 + term1
    naivebayes = np.argmax(sum, axis=1)
    if k>2:
        naivebayes = np.add(1, naivebayes)

    #print(naivebayes)


    return(naivebayes)

def print_top_words(params,vocab):

    n = len(vocab)
    param_mu = params[1]

    #print(param_mu)
    #print(param_mu[0])
    #print(param_mu[1])

    alpha1 = np.subtract(np.log(np.divide(param_mu[1], np.subtract(1,param_mu[1]))), np.log(np.divide(param_mu[0], np.subtract(1,param_mu[0]))))
    #print(alpha1)

    sortedalpha = np.argsort(alpha1)

    print("Top 20:")
    for i in range(20):
        index = sortedalpha[n - 1 - i]
        #print("%d. %s: %f" % (i + 1, vocab[index], alpha1[index]))
        print("%d. %s" % (i + 1, vocab[index]))

    print("Bottom 20:")
    for i in range(20):
        index = sortedalpha[i]
        #print("%d. %s: %f" % (i+1, vocab[index],  alpha1[index]))
        print("%d. %s" % (i + 1, vocab[index]))

def load_data():
    return loadmat('news.mat')

def load_vocab():
    with open('news.vocab') as f:
        vocab = [ x.strip() for x in f.readlines() ]
    return vocab

if __name__ == '__main__':
    news = load_data()

    # 20-way classification problem

    data = news['data']
    labels = news['labels'][:,0]
    testdata = news['testdata']
    testlabels = news['testlabels'][:,0]

    params = estimate_naive_bayes_classifier(data,labels)
    pred = predict(params,data) # predictions on training data
    testpred = predict(params,testdata) # predictions on test data

    print('20 classes: training error rate: %g' % np.mean(pred != labels))
    print('20 classes: test error rate: %g' % np.mean(testpred != testlabels))

    # binary classification problem

    indices = (labels==1) | (labels==16) | (labels==20) | (labels==17) | (labels==18) | (labels==19)
    data2 = data[indices,:]
    labels2 = labels[indices]
    labels2[(labels2==1) | (labels2==16) | (labels2==20)] = 0
    labels2[(labels2==17) | (labels2==18) | (labels2==19)] = 1
    testindices = (testlabels==1) | (testlabels==16) | (testlabels==20) | (testlabels==17) | (testlabels==18) | (testlabels==19)
    testdata2 = testdata[testindices,:]
    testlabels2 = testlabels[testindices]
    testlabels2[(testlabels2==1) | (testlabels2==16) | (testlabels2==20)] = 0
    testlabels2[(testlabels2==17) | (testlabels2==18) | (testlabels2==19)] = 1

    params2 = estimate_naive_bayes_classifier(data2,labels2)
    pred2 = predict(params2,data2) # predictions on training data
    testpred2 = predict(params2,testdata2) # predictions on test data

    print('2 classes: training error rate: %g' % np.mean(pred2 != labels2))
    print('2 classes: test error rate: %g' % np.mean(testpred2 != testlabels2))

    vocab = load_vocab()
    print_top_words(params2,vocab)
