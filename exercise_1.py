import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]

sample_x = np.zeros(shape=(50, 1))
sample_y = np.zeros(shape=(50, 1))
j = 0


for i in np.random.choice(442, 50):
    sample_x[j] = diabetes_X[i]
    sample_y[j] = diabetes_y[i]
    j = j + 1


def performLinearRegression(x, y, solverType):
    if solverType == "LeastSquares":
        findLeastSquaresSolution(x, y)

    elif solverType == "Ransac":
        findRansacSolution(x, y)
    else:
        print("Enter a valid value.")


def performGradientDescent(type):
    mse = 0
    if type == "LeastSquare":
        lr = linear_model.LinearRegression()
        lr.fit(sample_x, sample_y)
        estimatedY = lr.predict(sample_x)
        mse = mean_squared_error(sample_y, estimatedY)

    elif type == "Ransac":
        ransac = linear_model.RANSACRegressor()
        ransac.fit(sample_x, sample_y)
        estimatedY = ransac.predict(sample_x)
        mse = mean_squared_error(sample_y, estimatedY)

    return mse


def findLeastSquaresSolution(x, y):
    lr = linear_model.LinearRegression()
    lr.fit(x, y)
    diabetes_y_pred = lr.predict(diabetes_X)

    print(sample_x)
    print(sample_y)
    print(lr.score(x, y))

    ax1.plot(diabetes_X, diabetes_y_pred, color='blue', linewidth=4)

    rmse = "RMSE for Least Square: \n " + str(performGradientDescent("LeastSquare"))
    ax1.title.set_text(rmse)


def findRansacSolution(x, y):
    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)
    line_x = diabetes_X
    line_y_ransac = ransac.predict(diabetes_X)

    print(sample_x)
    print(sample_y)
    print(ransac.score(x, y))

    ax2.plot(line_x, line_y_ransac, color='blue', linewidth='4')

    rmse = "RMSE for Ransac: \n" + str(performGradientDescent("Ransac"))
    ax2.title.set_text(rmse)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Chapter 1')
performLinearRegression(sample_x, sample_y, "LeastSquares")
ax1.scatter(sample_x, sample_y, color='black')

performLinearRegression(sample_x, sample_y, "Ransac")
ax2.scatter(sample_x, sample_y, color='black')

plt.show()
