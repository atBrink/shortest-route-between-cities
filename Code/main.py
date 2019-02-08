import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial.ckdtree import cKDTree
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import time

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'

citiesRadius = 1
citiesDistRadius = {germanCities: 0.0025, hungaryCities: 0.005,
                    sampleCities: 0.08}

# Pick the county you want to travel in.
startCity = germanCities
STARTNODE = 1573 # 311
ENDNODE = 10584 # 702

# on = 1
# off = 0
showGraph = 0
fastOrNot = 1

functionSpeed = []
funcIndex = []

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
    mode1 = 'r'
    coord_list = []
    with open(filename, mode1) as citiesCoordinates:
        # TODO: change a indexing to split
        for coord in citiesCoordinates:
            a = coord[1:-2:]
            longitud, latitud = a.split(',')
            longitud, latitud = float(longitud), float(latitud)
            x = citiesRadius*(np.pi * latitud)/180
            y = citiesRadius*np.log(np.tan((np.pi/4)+(np.pi*longitud)/360))
            c = (x, y)
            coord_list.append(c)
    end = time.time()
    functionSpeed.append([read_coordinate_file.__name__,end - start])
    return np.array(coord_list)

# 2 - Creates a plot of the coordinates given
def plot_points(coord_list, indices, path, showGraph):
    '''
    Plots the cities locations

    PARAMETERS:
    -----------
    :param coord_list: numPy array, list of coordinates

    RETURNS:
    --------
    :return: plot of coordinates

    '''
    start = time.time()
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

    #for i, ind in enumerate(coord_list):
    #    ax.annotate(i, (coord_list[i,0], coord_list[i,1]))
    if showGraph == 1:
        plt.show()
    
    end = time.time()
    functionSpeed.append([plot_points.__name__, end - start])

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
    start = time.time()
    relations = []
    cost = []

    for city1, value1 in enumerate(coord_list):
        for city2 in range(city1, len(coord_list)):
            value2 = coord_list[city2]
            distance = np.linalg.norm(value1 - value2)

            if (np.less_equal(distance, radius) and distance != 0):
                relations.append((city1, city2))
                cost.append(distance**(9./10.))

    end = time.time()
    functionSpeed.append([construct_graph_connections.__name__, end - start])
    return np.array(relations), np.array(cost)


def construct_graph(indices, costs, N):
    '''
    Creates a tabular that contains the indices and the cost to travel between the two cities.

    PARAMETERS:
    -----------
    :param indices: a table that shows which cities that are within the radius of another city.
    :param costs: a table that shows the costs to travel from one city to another.
    :param N: number of nodes(cities)

    RETURNS:
    --------
    :return:a tabular with costs for certain distances between cities.
    '''
    start = time.time()
    M = N
    sparseMatrix = csr_matrix((costs, (indices[:, 0], indices[:, 1])),
                              shape=(M, N))
    # print(sparseMatrix, sep='')

    end = time.time()
    functionSpeed.append([construct_graph.__name__, end - start])
    return sparseMatrix


def cheapest_path(indices, startNode, endNode):
    '''
    Finally we are about to see which path that is the cheapest!! The method to solve the problem is the command
    Dijkstra, a function that is already in the python scipy.sparse.csgraph library.

    PARAMETERS:
    -----------
    :param indices: shows which cities that are related, and the cost to travel the distance
    :param startNode: the city to start in
    :param endNode: the city to end in.

    RETURNS:
    --------
    :return: the cheapest path from the startcity to the endcity, and the total cost of traveling it.
    '''
    start = time.time()
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
    
    end = time.time()
    functionSpeed.append([cheapest_path.__name__, end - start])
    return path[::-1], totalCost

def construct_fast_graph_connections(coord_list, radius):
    start = time.time()

    cost = []
    relations = []
    neighburs = []

    tree = cKDTree(coord_list)

    for i in coord_list:
        neighburs.append(tree.query_ball_point(i, radius))
    
    for i, cities in enumerate(neighburs):
        cityCon = []
        for neihboringCities in cities:
            if i != neihboringCities:
                cityCon = [i, neihboringCities]
                if cityCon[::-1] not in relations:
                    value1 = coord_list[cityCon[0]]
                    value2 = coord_list[cityCon[1]]
                    distance = np.linalg.norm(value1 - value2)

                    cost.append(distance**(9./10.))
                    relations.append(cityCon)
    end = time.time()
    functionSpeed.append([construct_fast_graph_connections.__name__, end - start])

    return np.array(relations), np.array(cost)


def main(city, startNode, endNode):
    '''
    This function is the one to rule them all. It's the function that joins the functions above and make it possible
    to show the desired answer.

    PARAMETERS:
    -----------
    :param city: the input-textfile that is read.
    :param startNode: the city to start in.
    :param endNode: the city to end in.

    RETURNS:
    --------
    :return:a plot that shows the cheapest way, an array that shows the path (all nodes) that is the cheapest,
    the cost of the path and how long the functions worked to return an answer.
    '''
    coord_list = read_coordinate_file(city) # contains coordinates for cities
    radius = citiesDistRadius[city] # grab radius from start node

    # fastOrNot = input("Want to run fast or regular graph connections? y/n \n")
    if fastOrNot == 1:
        indices, costs = construct_fast_graph_connections(coord_list, radius)
        graphConnectorMode = "Fast"
    else:
        indices, costs = construct_graph_connections(coord_list, radius)
        graphConnectorMode = "Regular"

    N = len(coord_list)
    sparseMatrix = construct_graph(indices, costs, N)
    path, totalCost = cheapest_path(sparseMatrix, startNode, endNode)

    # CREATE PLOT OF CITIES
    plot_points(coord_list, indices, path, showGraph)
 

    print('################################################')
    print('Data:')
    print('\t Chosen file: {}'.format(city.strip('.txt')))
    print('\t Chosen start city: {}'.format(startNode))
    print('\t Chosen end city: {}'.format(endNode))
    print('\t Cheapest path: {}'.format(path))
    print('\t Total Cost from start to finish: {:.4f}'.format(totalCost))
    print('----------------------------------------------')
    print('Time benchmark for the used functions:')
    print('\n'.join('\t {}: {:.4f}s'.format(i[0], i[1]) for k, i in enumerate(functionSpeed)))
    totTime = 0
    for i in functionSpeed:
        totTime = totTime + float(i[1])
    print('\t Total Time: {}s'.format(totTime))
    print('----------------------------------------------')
    print('Run info:')
    print('\t Graph Connection calc: {}'.format(graphConnectorMode))
    print('\t Show graph: {}'.format(showGraph))
    print('################################################')

    
main(startCity, STARTNODE, ENDNODE)
