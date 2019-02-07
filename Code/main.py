import numpy as np
# import scipy as sp
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'

citiesRadius = 1
citiesDistRadius = {germanCities: 0.0025, hungaryCities: 0.005,
                    sampleCities: 0.08}

# Pick the county you want to travel in.
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


# 2,5  - Creates a plot of the coordinates given
def plot_points(coord_list, indices):

    lines = []
    for i in indices:
        line = []

        # City 1
        x1 = coord_list[i[0], 0]
        y1 = coord_list[i[0], 1]

        # City 2
        x2 = coord_list[i[1], 0]
        y2 = coord_list[i[1], 1]

        # Connect the coordinates in one list.
        line = [(x1, y1), (x2, y2)]
        lines.append(line)

    lines_segments = LineCollection(lines)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    ax.add_collection(lines_segments)
    ax.set_xlim(min(coord_list[:, 0]), max(coord_list[:, 0]))
    ax.set_ylim(min(coord_list[:, 1]), max(coord_list[:, 1]))
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
    sparseMatrix = csr_matrix((costs, (indices[:, 0], indices[:, 1])),
                              shape=(M, N))
    print(sparseMatrix, sep='')

    return sparseMatrix


def main(city):
    radius = citiesDistRadius[city]
    indices, costs = construct_graph_connections(read_coordinate_file(city),
                                                 radius)
    # N = len(read_coordinate_file(city))
    plot_points(read_coordinate_file(city), indices)


main(startCity)
