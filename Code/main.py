import numpy as np
# import scipy as sp
from scipy.sparse import csr_matrix, csgraph
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'

citiesRadius = 1
citiesDistRadius = {germanCities: 0.0025, hungaryCities: 0.005,
                    sampleCities: 0.08}

# Pick the county you want to travel in.
startCity = hungaryCities


# 1 - Read Coordinate-file and convert the string-values to float-values,
# returns np.array of coordinates.
def read_coordinate_file(filename):
    '''
    Function to read the input file and change it from strings to numPy
     array with floats. It also calculates the coordinates which depend on
      the values in the input-file.

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
def plot_points(coord_list, indices, path):
    '''
    Plots the cities locations

    PARAMETERS:
    -----------
    :param coord_list: numPy array, list of coordinates

    RETURNS:
    --------
    :return: plot of coordinates

    '''

    lines = []
    cheapestPath = []

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
        if set(i).issubset(path):
            cheapestPath.append(line)
        else:
            lines.append(line)
    lines_segments = LineCollection(lines)
    cheapest_segment = LineCollection(cheapestPath, 
                                    linewidths=5,
                                    color='r')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    ax.add_collection(lines_segments)
    ax.add_collection(cheapest_segment)
    ax.set_xlim(min(coord_list[:, 0]), max(coord_list[:, 0]))
    ax.set_ylim(min(coord_list[:, 1]), max(coord_list[:, 1]))

    for i, ind in enumerate(coord_list):
        ax.annotate(i, (coord_list[i,0], coord_list[i,1]))
    plt.show()


# 3 - Creates a relation and cost array within a given radius of a city.
def construct_graph_connections(coord_list, radius):
    '''
    Calculates which cities that are within the given radius from the
    current city, and therefore a possible route.
    The distance between the cities is used to calculate the cost.

    PARAMETERS:
    -----------
    :param coord_list:  contains the coordinates of all cities in the read file.
    :param radius: a float that is given depending on which file that is
    being read.
    
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
    sparseMatrix = csr_matrix((costs, (indices[:, 0], indices[:, 1])),
                              shape=(M, N))
    # print(sparseMatrix, sep='')

    return sparseMatrix


def cheapest_path(indices, startNode, endNode):
    dist_matrix, predecessor = csgraph.dijkstra(csgraph=indices, directed=False, indices=startNode, return_predecessors=True)
    path = []
    path.append(endNode)
    for i in predecessor:
        prevNode = []
        if (predecessor[path[-1]] != -9999):
            prevNode = predecessor[path[-1]]
            path.append(prevNode)
        else:
            break
    totalCost = dist_matrix[endNode]
    
    return path[::-1], totalCost



def main(city, startNode, endNode):
    coord_list = read_coordinate_file(city) # contains coordinates for cities
    radius = citiesDistRadius[city] # grab radius from start node
    indices, costs = construct_graph_connections(coord_list, radius)
    N = len(coord_list)
    sparseMatrix = construct_graph(indices, costs, N)

    path, totalCost = cheapest_path(sparseMatrix, startNode, endNode)

    # CREATE PLOT OF CITIES
    plot_points(coord_list, indices, path)



STARTNODE = 60
ENDNODE = 553

main(startCity, STARTNODE, ENDNODE)
