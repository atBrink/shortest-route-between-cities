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
    '''
    Function to read the input file and change it from strings to numPy array with floats. It also calculates
    the coordinates which depend on the values in the input-file.

    PARAMETERS:
    ----------
    :param filename: string, the file to read

    RETURNS:
    --------
    :return: numPy array, containing coordinates
    '''
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
    '''
    Plots the cities locations

    PARAMETERS:
    -----------
    :param coord_list: numPy array, list of coordinates

    RETURNS:
    --------
    :return: plot of coordinates

    '''

    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=2)
    plt.show()


# 3 - Creates a relation and cost array within a given radius of a city.
def construct_graph_connections(coord_list, radius):
    '''
    Calculates which cities that are within the given radius from the current city, and therefore a possible route.
    The distance between the cities is used to calculate the cost.

    PARAMETERS:
    -----------
    :param coord_list: contains the coordinates of all cities in the read file.
    :param radius: a float that is given depending on which file that is beeing read.
    
    RETURNS:
    --------
    :return: an array that states which cities that are close to one another and an array that states the cost
    of traveling the distance.
    '''
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

# cheat sheet?
