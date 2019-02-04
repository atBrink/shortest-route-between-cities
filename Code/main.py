import numpy as np
# import scipy as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'

citiesRadius = 1
citiesDistRadius = {germanCities: 0.0025, hungaryCities: 0.005,
                    sampleCities: 0.08}

# STARTA HÄR YO!!!!!!!!!
startCity = sampleCities


# 1 - Read Coordinate-file and convert the string-values to float-values,
# returns np.array of coordinates.
def read_coordinate_file(filename):
    mode1 = 'r'
    coord_list = []
    with open(filename, mode1) as citiesCoordinates:

        for coord in citiesCoordinates:
            a = coord[1:-2:]
            longitud, latitud = a.split(',')
            longitud, latitud = float(longitud), float(latitud)
            x = citiesRadius*(np.pi * latitud)/180
            y = citiesRadius*np.log(np.tan((np.pi/4)+(np.pi*longitud)/360))
            c = (x, y)
            coord_list.append(c)

    return np.array(coord_list)


# 2 - Creates a plot of the coordinates given
def plot_points(coord_list):

    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=2)
    plt.show()


# 3 - Creates a relation and cost array within a given radius of a city.
def construct_graph_connections(coord_list, radius):
    relations = []
    cost = []

    for city1, value1 in enumerate(coord_list):
        for city2 in range(city1, len(coord_list)):
            value2 = coord_list[city2]
            distance = np.linalg.norm(value1 - value2)

            if (np.less_equal(distance, radius) and distance != 0):
                relations.append((city1, city2))
                cost.append(distance**(9./10.))

    return np.array(relations), np.array(cost)


def construct_graph(indices, costs, N):
    M = N

    test = csr_matrix((costs, indices[:, 0], indices[0, :]), shape=(M, N))
    print(test, sep='')


def main(city):
    radius = citiesDistRadius[city]
    indices, costs = construct_graph_connections(read_coordinate_file(city),
                                                 radius)
    N = len(read_coordinate_file(city))
    # print(read_coordinate_file(city))

    # plot_points(read_coordinate_file(city))
    construct_graph(indices, costs, N)


main(startCity)
