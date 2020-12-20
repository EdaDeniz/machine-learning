import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plotModel(n, coeffs, interval, dpi, color):

    coeffs = np.squeeze(coeffs)
    p = np.poly1d(coeffs)

    vectorx = np.arange(interval[0], interval[1], 1)
    vectory = p(vectorx)

    plt.scatter(vectorx, vectory, color=color)
    plt.plot(vectorx, vectory)


def exp1():
    vectorX = np.arange(0, 10, 1)
    vectorY = 3 * vectorX + 5

    vectorX = vectorX[:, np.newaxis]
    vectorY = vectorY[:, np.newaxis]

    poly_features = PolynomialFeatures(degree=0)
    x_poly = poly_features.fit_transform(vectorX)
    model = LinearRegression()
    model.fit(x_poly, vectorY)
    coeffs = model.coef_
    plotModel(0, coeffs, (0, 10), 1, 'green')

    poly_features = PolynomialFeatures(degree=1)
    x_poly = poly_features.fit_transform(vectorX)
    model = LinearRegression()
    model.fit(x_poly, vectorY)
    coeffs = model.coef_
    plotModel(1, coeffs, (0, 10), 1, 'red')

    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(vectorX)
    model = LinearRegression()
    model.fit(x_poly, vectorY)
    coeffs = model.coef_
    plotModel(2, coeffs, (0, 10), 1, 'blue')

    poly_features = PolynomialFeatures(degree=3)
    x_poly = poly_features.fit_transform(vectorX)
    model = LinearRegression()
    model.fit(x_poly, vectorY)
    coeffs = model.coef_
    plotModel(3, coeffs, (0, 10), 1, 'pink')

    plt.show()


def exp2(K):
    vectorX = np.random.uniform(0, 10, K)
    vectorP = np.random.uniform(15, 20, 10)
    vectorY = 3 * vectorX + 5
    vectorQ = 3 * vectorP + 5
    n1 = np.random.normal(0, 1, vectorX.shape)
    n2 = np.random.normal(0, 1, vectorX.shape)
    vectorY = (n1 + vectorY).reshape(-1,1)

    vectorQ = n2 + vectorQ

    poly_features = PolynomialFeatures(degree=0)
    x_poly = poly_features.fit_transform(vectorY)
    model = LinearRegression()
    model.fit(x_poly, vectorQ)
    coeffs = model.coef_
    plotModel(0, coeffs, (0, 10), 1, 'green')

    poly_features = PolynomialFeatures(degree=1)
    x_poly = poly_features.fit_transform(vectorY)
    model = LinearRegression()
    model.fit(x_poly, vectorQ)
    coeffs = model.coef_
    plotModel(0, coeffs, (0, 10), 1, 'red')

    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(vectorY)
    model = LinearRegression()
    model.fit(x_poly, vectorQ)
    coeffs = model.coef_
    plotModel(0, coeffs, (0, 10), 1, 'blue')

    poly_features = PolynomialFeatures(degree=3)
    x_poly = poly_features.fit_transform(vectorY)
    model = LinearRegression()
    model.fit(x_poly, vectorQ)
    coeffs = model.coef_
    plotModel(0, coeffs, (0, 10), 1, 'pink')

    plt.scatter(vectorX, vectorY, color='black', marker='x')
    plt.scatter(vectorP, vectorQ, color='black', marker='^')
    plt.show()


def main():
    K = 10
    exp1()
    exp2(K)
    print("hellooo")


if __name__ == "__main__":
    main()
