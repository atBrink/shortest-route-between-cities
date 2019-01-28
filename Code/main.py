import numpy as np
# import SciPy as sp
# import matplotlib as mpl

germanCities = "GermanyCities.txt"
hungaryCities = "HungaryCities.txt"
sampleCities = 'SampleCoordinates.txt'
mode1 = 'r'


def read_coordinate_file(filename):
    citiesCoordinates = open(filename, 'r')

    for coord in citiesCoordinates:
        a = np.array(coord)
        print(a)

    citiesCoordinates.close


read_coordinate_file("SampleCoordinates.txt")
