import numpy as np
import math

def setup():
    X = np.empty(shape=1001)
    Y = np.empty(shape=1001)

    epsilon = np.random.normal(0,1,size=1001)
    for n in range(1001):
        X[n] = -1*math.pi + n*(2*math.pi/1000)
        Y[n] = math.sin(3*X[n]/2) + epsilon[n]
        #print("X[n]=%.5f, Y[n]=%.5f" % (X[n],Y[n]))
    return X,Y

def ols_test_risk(X, Y, beta):
    n = X.shape[0]

    #testlabels = np.round(np.dot(X, beta))
    testlabels = np.dot(X,beta)

    count = 0

    for i in range(n):
        count += np.square(testlabels[i] - Y[i])

    testrisk = count/n
    return testrisk

if __name__ == '__main__':

    averagerisk = np.zeros(shape=20)

    for i in range(20):
        k=i+1
        print("k=%d" % k)
        for j in range(1000):
            X, labels = setup()
            exp_y = np.sin(3*X/2)

            data = np.empty(shape=(1001,k+1))
            for n in range(1001):
                for ki in range(k+1):
                    data[n][ki] = math.pow(X[n],ki)
                    #print("xi=%.5f, k=%d, xi,j=%.5f " % (X[n],ki,data[n][ki]))
                #print(data[n])
            OLSestimator = np.linalg.lstsq(data, labels, rcond=0)
            OLStestrisk = ols_test_risk(data,exp_y,OLSestimator[0])
            averagerisk[i] += OLStestrisk
        averagerisk[i] = averagerisk[i]/1000
        print("averagerisk = %.5f" % averagerisk[i])