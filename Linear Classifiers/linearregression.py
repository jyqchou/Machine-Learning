from scipy.io import loadmat
import numpy as np
from itertools import combinations

def load_data():
    return loadmat('wine.mat')

def estimate_olslinearreg_classifier(X,Y):
    n = X.shape[0]
    d = X.shape[1]

    OLS = np.linalg.lstsq(X, Y, rcond=0)
    beta = OLS[0]

    #print("Residuals: %d" %OLS[1])

    return beta

def ols_test_risk(X, Y, beta):
    n = X.shape[0]

    #testlabels = np.round(np.dot(X, beta))
    testlabels = np.dot(X,beta)

    count = 0

    for i in range(n):
        count += np.square(testlabels[i] - Y[i])
        #if testlabels[i] != Y[i]:
        #    count += 1

    testrisk = count/n
    return testrisk

def estimate_slp_classifier(X, Y, l, variables):

    argmin = np.empty(shape=(l+1))
    minrisk = -1

    for combo in combinations(range(1,12),l):
        iteration = np.append([0],combo)
        #print("Testing:")
        #print(iteration)

        data = X[:,iteration]

        OLSestimator = estimate_olslinearreg_classifier(data,Y)
        OLStestrisk = ols_test_risk(data,Y,OLSestimator)

        #print("Risk: %.5f" %OLStestrisk)

        if minrisk == -1 or OLStestrisk<minrisk:
            argmin = iteration
            minrisk = OLStestrisk

            #print("min found, update: ")
            #print(argmin)
            #print(minrisk)

    #print("argmin:")
    print(argmin)
    #print("smallest risk: %.5f" %minrisk)


    for i in range(l+1):
        s = argmin[i]
        if s == 0:
            print('%d: %.5f' % (s, OLSestimator[i]))
        else:
            print('%s: %.5f' % (variables[s-1], OLSestimator[i]))

    return (OLSestimator, argmin, minrisk)

def calculate_correlation(test, varindices, variables):
    vars = np.subtract(varindices[1:],1)


    corrMatrix = np.corrcoef(np.transpose(test)[1:])[vars]

    noabsCorr = np.array([sorted(corrMatrix[0],key=abs),sorted(corrMatrix[1],key=abs),sorted(corrMatrix[2],key=abs)])
    #print(noabsCorr)

    nonzeroVars = np.abs(corrMatrix)
    #print(nonzeroVars)


    sortedCorr = np.argsort(nonzeroVars, axis=1)
    #print(sortedCorr)

    top2 = sortedCorr[:,8:10]
    #print(top2)
    top2Corr = np.array([np.take(nonzeroVars[0],top2[0]),np.take(nonzeroVars[1],top2[1]),np.take(nonzeroVars[2],top2[2])])
    #print(top2Corr)

    for i in range(len(vars)):
        word = variables[vars[i]]
        mostCorrWord = variables[top2[i][1]]
        mostCorrWordCorr = top2Corr[i][1]
        secondCorrWord = variables[top2[i][0]]
        secondCorrWordCorr = top2Corr[i][0]
        print('%s - %s: %0.5f, %s: %0.5f' % (word, mostCorrWord, mostCorrWordCorr, secondCorrWord, secondCorrWordCorr))

    print('without absolute values:')
    for i in range(len(vars)):
        word = variables[vars[i]]
        mostCorrWord = variables[top2[i][1]]
        mostCorrWordCorr = noabsCorr[i][9]
        secondCorrWord = variables[top2[i][0]]
        secondCorrWordCorr = noabsCorr[i][8]
        print('%s - %s: %0.5f, %s: %0.5f' % (word, mostCorrWord, mostCorrWordCorr, secondCorrWord, secondCorrWordCorr))

    return

if __name__ == '__main__':
    wine = load_data()

    data = wine['data']
    labels = wine['labels'][:, 0]
    testdata = wine['testdata']
    testlabels = wine['testlabels'][:, 0]

    OLSestimator = estimate_olslinearreg_classifier(data,labels)
    OLStestrisk = ols_test_risk(testdata,testlabels,OLSestimator)
    print("OLS Test Risk: %.5f" % OLStestrisk)
    #OLStrainingrisk = ols_test_risk(data, labels, OLSestimator)
    #print("OLS Training Risk: %.5f" % OLStrainingrisk)

    variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    l=3
    SparseLinearPredictor = estimate_slp_classifier(data,labels,l,variables)
    argmin = SparseLinearPredictor[1]
    SLPTestRisk = ols_test_risk(testdata[:,argmin],testlabels,SparseLinearPredictor[0])
    print("SLP Test Risk: %.5f" % SLPTestRisk)

    calculate_correlation(testdata,argmin,variables)



