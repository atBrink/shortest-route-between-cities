import numpy as np
# import scipy as sp
# import matplotlib as mpl

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
        coord_list = np.concatenate([coord_list, a], axis=0)

    # Delete the row that we initialized with.
    coord_list = np.delete(coord_list, 0, axis=0)
    citiesCoordinates.close


# 2
def plot_points(coord_list):
    print("hej")


# 3
def construct_graph_connections(coord_list, radius):
    print("Hej")


read_coordinate_file("SampleCoordinates.txt")
