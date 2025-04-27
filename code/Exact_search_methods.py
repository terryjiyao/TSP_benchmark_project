import csv
import time
import pulp
import itertools
import numpy as np
from scipy.spatial import distance_matrix

### Get Coordinates and Distance Matrix ###

def get_coordinates_kaggle(file_path):
    coordinates = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            coordinates.append([float(row[0]), float(row[1])])
    coordinates = np.array(coordinates)
    return coordinates

def get_coordinates_tsplib(file_path):
    coordinates = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        inside_coords = False
        for row in reader:
            if row and row[0] == 'NODE_COORD_SECTION':  # Start reading coordinates
                inside_coords = True
                continue
            if row and row[0] == 'EOF':  # End of coordinates section
                inside_coords = False
                continue

            if inside_coords:
                # Extract coordinates (ignore the city number)
                x, y = row[0].split(" ")[1], row[0].split(" ")[2]
                coordinates.append((float(x), float(y)))
    return coordinates

def get_distance_matrix(coordinates):
    dist_matrix = distance_matrix(coordinates, coordinates)
    return dist_matrix

### Exact Search Methods ###

def Brute_Force(dist_matrix):
    st = time.time()
    n = len(dist_matrix)
    best_permutation = []
    min_distance = 10000000
    all_permutation = list(itertools.permutations(range(1, n)))
    for i in range(len(all_permutation)):
        route = list(all_permutation[i])
        route.insert(0, 0)
        route.append(0)
        total_distance = 0
        for j in range(len(route)-1):
            total_distance = total_distance + dist_matrix[route[j],route[j+1]]
        if total_distance < min_distance:
            min_distance = total_distance
            best_permutation = route

    t = time.time() - st
    return min_distance, best_permutation, t

def Held_Karp(dists):
    st = time.time()    
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = [0]
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)
    t = time.time() - st

    return opt, list(reversed(path)), t

def Linear_Programming(dist_matrix):
    st = time.time()
    n = len(dist_matrix)

    # Define the ILP problem
    problem = pulp.LpProblem("TSP", pulp.LpMinimize)

    # Decision variables: x[i][j] is 1 if the route from city i to j is used
    x = pulp.LpVariable.dicts('x', (range(n), range(n)), cat='Binary')

    # MTZ variables: u[i] helps to eliminate subtours (only needed for cities 1 to n-1)
    u = pulp.LpVariable.dicts('u', range(n), lowBound=0, upBound=n, cat='Continuous')

    # Objective function: minimize total travel distance
    problem += pulp.lpSum(dist_matrix[i][j] * x[i][j] for i in range(n) for j in range(n)), "TotalDistance"

    # Degree constraints: each city leaves and enters exactly once
    for i in range(n):
        problem += pulp.lpSum(x[i][j] for j in range(n) if j != i) == 1, f"Out_{i}"
        problem += pulp.lpSum(x[j][i] for j in range(n) if j != i) == 1, f"In_{i}"

    # MTZ Subtour elimination constraints (for cities 1 to n-1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                problem += u[i] - u[j] + n * x[i][j] <= n - 1, f"MTZ_{i}_{j}"

    # Solve the problem using PuLP's default solver (CBC)
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    solution_edges = [(i, j) for i in range(n) for j in range(n) if pulp.value(x[i][j]) == 1]

    next_city = {}
    for (i, j) in solution_edges:
        next_city[i] = j

    tour = [0]
    current = 0
    while True:
        current = next_city[current]
        tour.append(current)
        if current == 0:
            break

    opt = sum(dist_matrix[i, j] for i in range(n) for j in range(n) if pulp.value(x[i][j]) == 1)
    t = time.time() - st

    return opt, list(reversed(tour)), t