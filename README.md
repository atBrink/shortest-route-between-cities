# shortest-route-between-cities
Calculates and plots the shortest route between two cities in either Germany or Hungary. This code is written as a solution to the first computer assignment in the course "Object-oriented programming in Python" at Chalmers University of Technology, 2019.


It is possible to change starting-point and ending-point in the main.py file by chaning the integer in the "CITIESSTARTNODES" and "CITIESENDNODES" respectively. 

It is also possible to add more countries by adding a .txt file with the long/lat coordinates and editing the first part of the main.py file.
Mainly:
- Add a new variable *country*cities that point to the new country coordiante file.
- edit if-else to include the new country
- Add an entry for the new country in the three variables CITIESSTARTNODES, CITIESENDNODES, CITIESDISTRADIUS.
