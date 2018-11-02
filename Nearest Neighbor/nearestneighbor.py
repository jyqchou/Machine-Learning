import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt

def nearestNeighbor(training, labels, test):

    n = len(training)
    d = len(training[0])
    t = len(test)

    trainingSquared = np.empty(shape=n)
    for x in range(n):
        trainingSquared[x] = np.dot(np.transpose(training[x]), training[x])
    #print(trainingSquared)

    testSquared = np.empty(shape=t)
    for x in range(t):
        testSquared[x] = np.dot(np.transpose(test[x]), test[x])
    #print(testSquared)

    cross = np.matmul(training, np.transpose(test))
    #print(cross)

    nnLabels = np.empty(shape=t)


    #take each test feature vector
    for n1 in range(t):
        #print(n1)
        distances = np.empty(shape = n)

        #calculate distance between test and each training
        for n2 in range(n):
            distance = trainingSquared[n2] - 2*cross[n2][n1] + testSquared[n1]
            #print(distance)
            distances[n2] = distance

        #get nearest neighbor in training set
        nn = np.argmin(distances)

        #assign corresponding label
        nnLabels[n1] = labels[nn]

    return nnLabels



if __name__ == '__main__':
    ocr = loadmat('ocr.mat')

    '''
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.imshow(ocr['data'][0].reshape((28,28)), cmap=cm.gray_r)
    plt.show()
    '''

    #training feature vectors: data, labels
    #test feature vectors: testdata testlabels

    data = np.array(ocr['data'].astype('float'))
    labels = np.array(ocr['labels'].astype('float'))
    testdata = np.array(ocr['testdata'].astype('float'))
    testlabels = np.array(ocr['testlabels'].astype('float'))

    #nearest neighbor classifier on all of training data
    #assignedNN = nearestNeighbor(data, labels, testdata)

    #output model's labels for test data
    #print(assignedNN)

    testErrorRate = np.empty(shape=(4, 10), dtype=float)
    trainingDataSizes = [1000, 2000, 4000, 8000]


    for i in range(10):
        print('Trial # %d' % (i+1))
        for n in trainingDataSizes:
            print('n = %d' % n)
            sel = random.sample(range(60000), n)
            data = np.array(ocr['data'][sel].astype('float'))
            labels = np.array(ocr['labels'][sel].astype('float'))

            #print(data)
            #print(labels)

            assignedNN = nearestNeighbor(data, labels, testdata)
            #print(assignedNN[:10])

            error = 0.0
            for x in range(len(testdata)):
                if assignedNN[x] != testlabels[x]:
                    error += 1
            #print(error)
            errorRate = error/len(testdata)
            print('For n = %d, Test Error Rate = %.4f' % (n, errorRate))

            if n == 1000:
                testErrorRate[0][i] = errorRate
            elif n == 2000:
                testErrorRate[1][i] = errorRate
            elif n == 4000:
                testErrorRate[2][i] = errorRate
            elif n == 8000:
                testErrorRate[3][i] = errorRate

    print("TestErrorRate:")
    print(testErrorRate)


    '''
    for a in range(4):
        for b in range(10):
            testErrorRate[a][b] = random.random()
    print(testErrorRate)
    '''

    testErrorRateMean = np.mean(testErrorRate, axis=1)
    testErrorRateStd = np.std(testErrorRate, axis=1)
    #print(testErrorRateMean)
    #print(testErrorRateStd)

    plt.figure()
    plt.title('Learning Curve of Nearest Neighbor Classifier')
    plt.xlabel('Size of Training Data (n)')
    plt.ylabel('Test Error Rate')
    plt.errorbar(trainingDataSizes, testErrorRateMean, testErrorRateStd,
                 fmt='rs--', marker='s', ecolor='k', capsize=5, capthick=0.5)
    for xy in zip(trainingDataSizes, testErrorRateMean):
        plt.annotate('(%s, %.4f)' % xy, xy=xy, textcoords='data')

    plt.show()