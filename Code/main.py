import numpy as np
# import scipy as sp
# import matplotlib as mpl

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'


def read_coordinate_file(filename):
    mode1 = 'r'
    citiesCoordinates = open(filename, mode1)
    compList = np.array([])
    print(np.shape(compList))
    for coord in citiesCoordinates:
        a = coord[1:-2:]
        a = a.split(',')
        a = np.array([float(a[0]), float(a[1])])
        #compList = np.append(compList, a, axis=1)
        compList = np.concatenate((compList, a), axis=1)
    print(compList)

    citiesCoordinates.close


read_coordinate_file("SampleCoordinates.txt")
