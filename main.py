import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial import distance
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import time
import math
import os

pathToCityData= os.getcwd()+'/data/'
GERMANCITIES = pathToCityData+"GermanyCities.txt"
HUNGARYCITIES = pathToCityData+"HungaryCities.txt"
SAMPLECITIES = pathToCityData+'SampleCoordinates.txt'

##########################################################################

# Pick the county you want to travel in.
print('Choose which city to travel in: \n Germany: 0 \n Hungary: 1')
chosenCity = input('City:')
if chosenCity == "0":
    STARTCITY = GERMANCITIES
elif chosenCity == "1":
    STARTCITY = HUNGARYCITIES
else:
    STARTCITY = SAMPLECITIES

# Toggle for graph and different graph connection calculations.
# on = 1
# off = 0
showGraph = 1
fastOrNot = 1

###############################################################################
CITIESRADIUS = 1
CITIESSTARTNODES = {GERMANCITIES: 1573, HUNGARYCITIES: 311,
                    SAMPLECITIES: 0}
CITIESENDNODES = {GERMANCITIES: 10584, HUNGARYCITIES: 702,
                  SAMPLECITIES: 5}
CITIESDISTRADIUS = {GERMANCITIES: 0.0025, HUNGARYCITIES: 0.005,
                    SAMPLECITIES: 0.08}


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
    start = time.time()
    mode1 = 'r'     # Read from fily only
    coord_list = []

    # open coordinate file and read line for line, extracting from the strings
    # the longitud and latituds as floats, then calculate the corresponding x
    # and y values of the cities and save to a list.
    with open(filename, mode1) as citiesCoordinates:
        for coord in citiesCoordinates:
            coord = coord.strip("{")
            coord = coord.rstrip("} \n")
            longitud, latitud = coord.split(',', 1)
            longitud, latitud = float(longitud), float(latitud)

            # Convert long/lat using the Mercator projection to obtain the
            # coordinates in xy-format:
            x = CITIESRADIUS*(math.pi * latitud)/180
            y = CITIESRADIUS*math.log(math.tan((math.pi/4) +
                                               (math.pi*longitud)/360))
            c = (x, y)
            coord_list.append(c)
    end = time.time()
    functionSpeed.append([read_coordinate_file.__name__, end - start])
    return np.array(coord_list)


def plot_points(coord_list, indices, path, showGraph):
    '''
    Plots the cities locations

    PARAMETERS:
    -----------
    :param coord_list: numPy array, list of coordinates
    :param indices: numPy array containing the cities and their neighburs
    :param path: list containing the route to plot
    :param showGraph: 1 or 0 depending on wether to show graph

    RETURNS:
    --------
    :return: Shows a plot of the given coordinates and the route given in path

    '''
    start = time.time()
    lines = []
    cheapestPath = []

    # Loops over every city connection from input and splits them into two
    #  cities.  If the connection is found in the cheapest path we add it to
    # cheapestPath list. if not, we add the connection to the lines list.
    for city_connection in indices:
        line = []
        city1 = [coord_list[city_connection[0], 0],
                 coord_list[city_connection[0], 1]]
        city2 = [coord_list[city_connection[1], 0],
                 coord_list[city_connection[1], 1]]

        line = [city1, city2]

        if set(city_connection).issubset(path):
            cheapestPath.append(line)
        else:
            lines.append(line)

    lines_segments = LineCollection(lines, linewidths=0.8, color='gray',
                                    label="Connections")
    cheapest_segment = LineCollection(cheapestPath, linewidths=2, color='red',
                                      label="Cheapest Path")

    fig = plt.figure()
    ax = fig.gca()
    ax.axis("equal")
    ax.scatter(coord_list[:, 0], coord_list[:, 1], s=15, facecolor='red',
               label="Cities")
    ax.add_collection(lines_segments)
    ax.add_collection(cheapest_segment)
    ax.legend()

    end = time.time()
    functionSpeed.append([plot_points.__name__, end - start])

    if showGraph == 1:
        plt.show()


def construct_graph_connections(coord_list, radius):
    '''
    Calculates which cities that are within the given radius from the
    current city, and therefore a possible route.
    The distance between the cities is used to calculate the cost.

    PARAMETERS:
    -----------
    :param coord_list: numPy array contains the coordinates of all cities in
                       the read file.
    :param radius: a float that is given depending on which file that is
    being read.

    RETURNS:
    --------
    :return: a numPy array that states which cities that are close to one
            another and an array that states the cost of traveling the
            distance.
    '''
    start = time.time()
    relations = []
    cost = []
    # Loop over the cities and compare between every other city whether it is
    # within the given radius or not.  If the cities have a relation within the
    # given values save to list "relations and calculate the cost of travel
    # between them and save it to a list "cost".
    for city1, value1 in enumerate(coord_list):
        for city2, value2 in enumerate(coord_list):
            if city1 <= city2:
                continue
            diff = value1 - value2
            distance = math.hypot(*diff)
            if distance < radius and city1 != city2:
                relations.append([city1, city2])
                cost.append(distance**(9./10.))
    end = time.time()
    functionSpeed.append([construct_graph_connections.__name__, end - start])
    return np.array(relations), np.array(cost)


def construct_graph(indices, costs, N):
    '''
    Creates a tabular that contains the indices and the cost to travel between
    the two cities.

    PARAMETERS:
    -----------
    :param indices: numPy array, a table that shows which cities that are
                    within the radius of another city.
    :param costs: numpy Array, a table that shows the costs to travel from
                  one city to another.
    :param N: int, number of nodes(cities)

    RETURNS:
    --------
    :return:a tabular with costs for certain distances between cities.
    '''
    start = time.time()

    # We want the matrix rows and columns to be equal in order to show every
    # relation between every node. N = M
    M = N
    sparseMatrix = csr_matrix((costs, (indices[:, 0], indices[:, 1])),
                              shape=(M, N))

    end = time.time()
    functionSpeed.append([construct_graph.__name__, end - start])
    return sparseMatrix


def cheapest_path(indices, startnode, endNode):

    '''
    Finally we are about to see which path that is the cheapest!!  The method
    to solve the problem is the command Dijkstra, a function that is already
    in the python scipy.sparse.csgraph library.

    PARAMETERS:
    -----------
    :param indices: numpy array, shows which cities that are related,
                     and the cost to travel the distance.
    :param startnode: int, the city to start in
    :param endNode: int, the city to end in.

    RETURNS:
    --------
    :return: The cheapest path from the startcity to the endcity, and the
             total cost of traveling it.
    '''
    start = time.time()

    # We do not need to take directed paths into account,
    # since we only want the cheapest path between two nodes.
    # The csgrapth parameter is a matrix containing the relations between
    # every city.  (if two cities is within the given radius) indices parameter
    # is the sort of selection parameter, it lets us tell the function to only
    # compute the path we choose.  This will give us a matrix containing the
    # distance between nodes and a matrix with the nodes preceding the
    # "current node"
    dist_matrix, predecessor = csgraph.dijkstra(csgraph=indices,
                                                directed=False,
                                                indices=startnode,
                                                return_predecessors=True)
    path = []
    path.append(endNode)

    # We start at the endNode since the predecessor matrix contains the
    # previous node to get to the current node. We start at the endnode and
    # work backwards "through" the matrix until we end up at the startnode.
    for i in predecessor:
        prevNode = predecessor[path[-1]]
        path.append(prevNode)

        # When we end up on the startnode, add it and break the loop.
        if (predecessor[path[-1]] == startnode):
            path.append(startnode)
            break

    totalCost = dist_matrix[endNode]

    end = time.time()
    functionSpeed.append([cheapest_path.__name__, end - start])

    # Since the path is appended "backwards" due to the nature of the
    # predecessor matrix we reverse it in the return statement.
    return path[::-1], totalCost


def construct_fast_graph_connections(coord_list, radius):
    '''
    Calculates which cities that are within the given radius from the
    current city, and therefore a possible route using CKDtree method
    and querying for neighburs per node.
    The distance between the cities is used to calculate the cost.

    PARAMETERS:
    -----------
    :param coord_list: numpyarray, contains the coordinates of all cities in
                        the read file.
    :param radius: float that is given depending on which file that is being
                    read.

    RETURNS:
    --------
    :return: an array that states which cities that are close to one another
            and an array that states the cost of traveling the distance.
    '''
    start = time.time()

    cost = []
    relations = []
    # Create a index of the cooridnates to enable faster processing and
    # filter the coordinates so we do not need to compute every relation
    tree = cKDTree(coord_list)
    cityNeighburs = tree.query_ball_point(coord_list, radius)

    # Loop over the queried Tree index of the coordinates and compare the
    # cities to the cities around it (closecities).  If the cities are not the
    # same we add the pair as a relation.  The if-statement removes the
    # self-relation (city1 = city2 is not relevant)
    for city1, closeCities in enumerate(cityNeighburs):
        for city2 in closeCities:
            if city1 != city2:
                relations.append([city1, city2])

    # Convert the relations to a numpy-array and sort it and lastly the unique
    # method is used to remove all duplicates.  The relation (city1, city2)
    # and (city2, city1) describes the same thing but is not removed unless
    # unique is called.
    relations = np.array(relations)
    relations = np.sort(relations)
    relations = np.unique(relations, axis=0)

    # Calculating the cost between the cities
    city1 = coord_list[relations[:, 0]]
    city2 = coord_list[relations[:, 1]]
    cost = (city1 - city2)**2
    cost = np.sum(cost, axis=1)
    cost = np.sqrt(cost)
    cost = cost**(9/10)

    end = time.time()
    functionSpeed.append([construct_fast_graph_connections.__name__,
                         end - start])
    return relations, cost


def main(city):
    '''
    This function is the one to rule them all. It's the function that joins
    the functions above and make it possible to show the desired answer.

    PARAMETERS:
    -----------
    :param city: the input-textfile that is read.

    RETURNS:
    --------
    :return: a plot that shows the cheapest way, an array that shows the path
             (all nodes) that is the cheapest, the cost of the path and how
             long the functions worked to return an answer.
    '''
    coord_list = read_coordinate_file(city)  # contains coordinates for cities
    radius = CITIESDISTRADIUS[city]  # grab radius from start node
    startnode = CITIESSTARTNODES[city]
    endnode = CITIESENDNODES[city]

    if fastOrNot == 1:
        indices, costs = construct_fast_graph_connections(coord_list, radius)
        graphConnectorMode = "Fast"
    else:
        indices, costs = construct_graph_connections(coord_list, radius)
        graphConnectorMode = "Regular"

    N = len(coord_list)
    sparseMatrix = construct_graph(indices, costs, N)
    path, totalCost = cheapest_path(sparseMatrix, startnode, endnode)

    # CREATE PLOT OF CITIES
    plot_points(coord_list, indices, path, showGraph)

    # Prints information to terminal
    print('################################################')
    print('Data:')
    print('\t Chosen file: {}'.format(city.strip('.txt')))
    print('\t Chosen start city: {}'.format(startnode))
    print('\t Chosen end city: {}'.format(endnode))
    print('\t Cheapest path: {}'.format(path))
    print('\t Total Cost from start to finish: {:.4f}'.format(totalCost))
    print('----------------------------------------------')
    print('Time benchmark for the used functions:')
    print('\n'.join('\t {}: {:.4f}s'.format(i[0], i[1]) for k,
                    i in enumerate(functionSpeed)))
    totTime = 0
    for i in functionSpeed:
        totTime = totTime + float(i[1])
    print('\t Total Time: {}s'.format(totTime))
    print('----------------------------------------------')
    print('Run info:')
    print('\t Graph Connection calc: {}'.format(graphConnectorMode))
    print('\t Show graph: {}'.format(showGraph))
    print('################################################')


# lists to contain the time it takes to run each function.
functionSpeed = []
funcIndex = []

# Run program
main(STARTCITY)
