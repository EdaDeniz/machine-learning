import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.utils import random


def produceCirclePoints(center, radius, numOfPoints):
    randomR = np.random.randint(0, radius, numOfPoints)
    theta = np.random.uniform(0, 2 * np.pi, numOfPoints)
    x = randomR * np.cos(theta)
    y = randomR * np.sin(theta)
    x = x + center[0]
    y = y + center[1]
    return x, y


def prepareData():

    sample_x = np.arange(0, 1000, 1)

    raw_y = 3 * sample_x + 5

    numberOfSamples = raw_y.shape[0]

    noise = np.random.normal(0, 100, size=sample_x.shape)

    sample_y = raw_y + noise
    noise_2 = 2000 * np.random.randn(numberOfSamples) - 1000

    idx = np.random.randint(0, numberOfSamples, numberOfSamples // 2)

    noise_2[idx] = 0
    sample_y = raw_y + noise_2

    small_x = np.arange(-100, 100, 1)

    return sample_x, sample_y


def performLinearRegression(x, y, solverType):
    answer = 0
    if solverType == "LeastSquares":
        answer = findLeastSquaresSolution(x, y)
    elif solverType == "Ransac":
        answer = findRansacSolution(x, y)
    else:
        print("Enter a valid value.")
    return answer


def findLeastSquaresSolution(x, y):
    lr = linear_model.LinearRegression()
    lr.fit(x, y)
    return lr


def findRansacSolution(x, y):
    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)
    return ransac


def main():
    sample_x, sample_y = prepareData()
    #sample_x = sample_x.reshape((len(sample_x), 1))

    new_x, new_y = produceCirclePoints((100,7000),50,300)
    final_x = np.append(sample_x, new_x)
    final_y = np.append(sample_y, new_y)
    final_x = final_x.reshape((len(final_x), 1))

    lsmodel = performLinearRegression(final_x, final_y, "LeastSquares")
    ranmodel = performLinearRegression(final_x, final_y, "Ransac")


    ls_y = lsmodel.predict(final_x)
    rns_y = ranmodel.predict(final_x)

    fig, ax = plt.subplots()
    ax.scatter(final_x, final_y, color='black')
    ax.scatter(final_x, ls_y, color='blue', label="LS")
    ax.scatter(final_x, rns_y, color='red', label="Ransac")
    ax.legend(loc='upper left', frameon=False);
    plt.show()



    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle('Chapter 1')
    #ax1.scatter(sample_x, sample_y, color='black')
    #ax2.scatter(sample_x, sample_y, color='black')
    #plt.show()


if __name__ == "__main__":
    main()
