# AMCS6035 Group Project - TSP Benchmarking

This tiny project aims for testing 3 exact and 3 heurstic algorithm designed for TSP problem.

What is TSP problem?
The TSP can be formulated as an integer linear program or a graph problem.
https://en.wikipedia.org/wiki/Travelling_salesman_problem


## Exact Algorithms
### Brute Force

Reference: https://github.com/AbrarJahin/travelling-salesman-problem-brute-force


### Dynamic Programming gi algorithms

Reference: https://github.com/ishanjogalekar/TSP-using-Dynamic-Programming-?tab=readme-ov-file

### Integer linear programming

Reference: https://github.com/Arthod/LP-tsp-gurobi 

## Heuristic Algorithms
### Nearest Neighbor

Reference: 
https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm
https://github.com/chingisooinar/KNN-python-implementation
https://gist.github.com/mkocabas/feb9c4431e552aa33ffe2dc95d58c76c

### Lin–Kernighan K-opt

Reference: https://en.wikipedia.org/wiki/Lin%E2%80%93Kernighan_heuristic
https://github.com/kikocastroneto/lk_heuristic

### Christoph Deschauer Algorithm

Reference: https://github.com/Retsediv/ChristofidesAlgorithm

## Results

Result route for different hueristic search algorithms.
![route](/fig/heuristic.route.png)

Time compliexity for different hueristic search algorithms.
![compliexity](/fig/heuristic.timecomplexity2.png)

Stability of start node for different hueristic search algorithms.
![startnode](/fig/heuristic.stability.randomstart.png)

Stability of path node for different hueristic search algorithms.
![pathnode](/fig/heuristic.stability.randomnodes.png)

Result route for different exact search algorithms.
![route](/fig/exact.route.png)

## Code

Example code for exact search
```python
import numpy as np
import Exact_search_methods as esm
import matplotlib.pyplot as plt

file_path = '../dataset/Github/tiny.csv'
coordinates = esm.get_coordinates_kaggle(file_path)
dist_matrix = esm.get_distance_matrix(coordinates)
o1, p1, t1 = esm.Brute_Force(dist_matrix)
print("Method: Brute_Force, ", "Time: ", t1, ", Path Distance: ", o1)
o2, p2, t2 = esm.Held_Karp(dist_matrix)
print("Method: Held Karp, ", "Time: ", t2, ", Path Distance: ", o2)
o3, p3, t3 = esm.Linear_Programming(dist_matrix)
print("Method: Linear Programming, ", "Time: ", t3, ", Path Distance: ", o3)
```

Example code for heuristic search
```python
import numpy as np
import Heuristic_search_methods as hsm
import matplotlib.pyplot as plt

file_path = '../dataset/Github/small.csv'
coordinates = hsm.get_coordinates_kaggle(file_path)
dist_matrix = hsm.get_distance_matrix(coordinates)
o1, p1, t1 = hsm.Nearest_Neighbor(dist_matrix)
print("Method: Nearest Neighbor, ", "Time: ", t1, ", Path Distance: ", o1)
o2, p2, t2 = hsm.Lin_Kernighan_Flip(dist_matrix)
print("Method: Lin Kernighan Flipping, ", "Time: ", t2, ", Path Distance: ", o2)
o3, p3, t3 = hsm.Lin_Kernighan_k_Opt(dist_matrix, 2)
print("Method: Lin Kernighan 2-Opt, ", "Time: ", t3, ", Path Distance: ", o3)
o4, p4, t4 = hsm.Christoph_Deschauer(dist_matrix)
print("Method: Christoph Deschauer, ", "Time: ", t4, ", Path Distance: ", o4)
```

## Reference
Traveling salesman problem: Theory and applications[M]. BoD–Books on Demand, 2010.

Neoh A, Chen H, Chase C. An Evaluation of the Traveling Salesman Problem[J]. 2020.

Kitjacharoenchai P, Ventresca M, Moshref-Javadi M, et al. Multiple traveling salesman problem with drones: Mathematical model and heuristic approach[J]. Computers & Industrial Engineering, 2019, 129: 14-30.
