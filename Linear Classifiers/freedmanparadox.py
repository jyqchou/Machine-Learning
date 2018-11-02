from scipy.io import loadmat
import numpy as np

def load_data():
    return loadmat('freedman.mat')

def calculate_correlation(X, Y):
    n = X.shape[0]
    d = X.shape[1]
    corr = np.divide(np.sum(np.multiply(np.transpose(X),Y),axis=1),n)
    #print(corr.shape)
    return corr

def screen_features(X,rho):
    n = X.shape[0]
    d = X.shape[1]
    corr_cuttoff = 1.75/np.sqrt(n)
    abs_corr = np.abs(rho)

    Xj = X[:,abs_corr>corr_cuttoff]
    print(Xj.shape)
    return Xj

def screen_i_features(X):
    Xi = X[:,:75]
    print(Xi.shape)
    return Xi

def calculate_emprisk(X,Y,beta):
    n = Y.shape
    risk = np.sum(np.square(np.subtract(np.dot(X, beta), Y))) / n
    return risk

def simulate():
    for d in [5, 10, 25, 50, 75, 100, 200, 500, 1000]:
        for n in [10, 25, 50, 75, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
            if (n<=d):
                continue
            X = np.random.normal(size=(n,d))
            Y = np.random.normal(size=n)

            OLSestimator = np.linalg.lstsq(X,Y,rcond=0)
            print('d=%d, n=%d, risk=%.5f' % (d,n,OLSestimator[1]/n))


if __name__ == '__main__':
    freedman = load_data()

    data = freedman['data']
    labels = freedman['labels'][:, 0]

    #print(data.shape[0])
    #print(data.shape[1])
    #print(labels.shape)

    correlation = calculate_correlation(data, labels)

    Xj = screen_features(data,correlation)

    OLSestimator = np.linalg.lstsq(Xj,labels,rcond=0)
    #betaJrisk = calculate_emprisk(Xj,labels,OLSestimator[0])
    #print(betaJrisk)
    print('residuals = %.5f' %OLSestimator[1])
    print(OLSestimator[1]/Xj.shape[0])

    Xi = screen_i_features(data)
    OLSIestimator = np.linalg.lstsq(Xi,labels,rcond=0)
    betaIrisk = calculate_emprisk(Xi,labels,OLSIestimator[0])
    print(betaIrisk)

    simulate()