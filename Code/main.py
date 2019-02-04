import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'

citiesRadius = 1


# 1
def read_coordinate_file(filename):
    mode1 = 'r'
    citiesCoordinates = open(filename, mode1)
    coord_list = np.array([0, 0], ndmin=2)

    for coord in citiesCoordinates:
        a = coord[1:-2:]
        a = a.split(',')
        a = np.array([float(a[0]), float(a[1])], ndmin=2)
        x = citiesRadius*(np.pi * a[0, 1])/180
        y = citiesRadius*np.log(np.tan((np.pi/4)+(np.pi*a[0, 0])/360))

        a[0, 0] = x
        a[0, 1] = y
        coord_list = np.concatenate([coord_list, a], axis=0)

    # Delete the row that we initialized with.
    coord_list = np.delete(coord_list, 0, axis=0)
    print(coord_list)
    citiesCoordinates.close
    return coord_list


# 2
def plot_points(coord_list):
    read_coordinate_file("SampleCoordinates.txt")
    print(coord_list)
    x=[]
    y=[]
    for col in coord_list:
        x.append(col[0])
    print(x)
    for rad in coord_list:
        y.append(rad[1])
    print(y)
    # y = coord_list(1::2)
    plt.scatter(coord_list[:, 0], coord_list[:, 1])
    print("hej")
    plt.show()
    # använd två parametrar i numrate(?)


# 3
def construct_graph_connections(coord_list, radius):
    for index, value in enumerate(coord_list, 1):
        print("Index", index)
        print("Value", value)
    print("Hej")


def main(city):
    # plot_points(read_coordinate_file(city))
    # print(read_coordinate_file(city))
    # read_coordinate_file(city)
    construct_graph_connections(read_coordinate_file(city), 0.08)


main(sampleCities)

