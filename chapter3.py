from matplotlib.patches import Ellipse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import math
from math import pi


def generateStandartEllipse(centerx,centery,xaxis,yaxis,numberOfPoints):
    t = np.linspace(0, 2 * pi, numberOfPoints)
    xpoints = centerx+xaxis*np.cos(t)
    ypoints = centery+yaxis*np.sin(t)

    return xpoints, ypoints


def rotatePoints(x,y,angle):

    angle = math.radians(angle)
    qx = math.cos(angle) * x - math.sin(angle) * y
    qy = math.sin(angle) * x + math.cos(angle) * y
    return qx, qy


def generateEllipse(x,y,xaxis,yaxis,numberOfPoints,angle):
    xpoints, ypoints = generateStandartEllipse(x,y,xaxis,yaxis,numberOfPoints)
    qx, qy = rotatePoints(xpoints, ypoints,angle)
    return qx,qy

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def experiment1():

    x1,y1 = generateEllipse(15,20,10,15,50,90)
    x2,y2 = generateEllipse(15,20,5,3,50,90)
    vectorx = []
    vectory = []
    X0 = np.zeros(100)
    X1 = np.zeros(100)
    for i in range(len(x1)):
        vectorx.append([x1[i],y1[i]])
        X0[i]=x1[i]
        X1[i]=y1[i]
        vectory.append(1)
    for i in range(len(x2)):
        vectorx.append([x2[i],y2[i]])
        X0[i+50]=x2[i]
        X1[i+50]=y2[i]
        vectory.append(2)




    model = SVC(kernel='linear')
    model.fit(vectorx,vectory)

    fig, ax = plt.subplots()
    # title for the plots
    title = ("Decision surface of linear SVC ")
    # Set-up grid for plotting.
    #X0, X1 = vectorx[:, 0], vectorx[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=vectory, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()
    return

def experiment2():
    x=10
    y=10
    angle=50
    x1, y1 = generateEllipse(x, y, 10, 15, 50, angle)
    x2, y2 = generateEllipse(x, y, 5, 3, 50, angle)
    vectorx = []
    vectory = []
    for i in range(len(x1)):
        vectorx.append([x1[i], y1[i]])
        vectory.append(1)
    for i in range(len(x2)):
        vectorx.append([x2[i], y2[i]])
        vectory.append(2)

    model = SVC(kernel='linear')
    model.fit(vectorx, vectory)

    plt.scatter(x1, y1, color='green')
    plt.scatter(x2, y2, color='red')
    plt.show()
    return

def main():
    experiment1()
    #experiment2()

if __name__ == "__main__":
    main()