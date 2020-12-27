import numpy as np
from sklearn.utils import random
import math


def generateStandardEllipse(center, semiX, semiY, nop):
    print()


def rotatePoints(XY , angle):
    X, Y = XY

    rx = math.cos(angle) * X - math.sin(angle) * Y
    ry = math.sin(angle) * X + math.cos(angle) * Y

    return rx, ry


def generateEllipse(XY_center, a, b, num_points, angle):
    points_array = []
    for i in range(num_points):

        radius = a * b / np.sqrt((b*np.cos(angle))**2 + (a*np.sin(angle))**2)
        random_radius = radius * np.sqrt(random.random())

        points_array.append((random_radius * np.cos(angle) , random_radius * np.sin(angle)))

    return points_array

def experiment1():
    print()

