import csv
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
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

### Calculate Tour Path ###

def Calculate_Tour_Length(hamPath, dist_matrix):
    tourLength=sum(dist_matrix[hamPath[0:-1], hamPath[1:len(hamPath)]])
    tourLength+=dist_matrix[hamPath[-1],hamPath[0]]
    return tourLength

### Heurstic Algorithms for TSP Problems ###

def Nearest_Neighbor(dist_matrix, start_node = 0):
    st = time.time()
    n = dist_matrix.shape[0]

    visited = [start_node]  
    route = [start_node]
    current_point = start_node

    for _ in range(1, n):
        nearest_distance = float('inf')
        nearest_point = -1
        for j in range(n):
            if j not in visited:
                dist = dist_matrix[current_point, j]
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_point = j
        visited.append(nearest_point)
        route.append(nearest_point)
        current_point = nearest_point

    # return to the initial point
    route.append(start_node)
    total_distance = Calculate_Tour_Length(route, dist_matrix)
    t = time.time() - st 

    return total_distance, route, t


def Lin_Kernighan_Flip(dist_matrix, start_node = 0):
    st = time.time()
    n = len(dist_matrix)

    # Generate initial tour starting from start_node
    optlist = [start_node] + [i for i in range(n) if i != start_node]
    improvement = True

    while improvement:
        improvement = False  # Set to False at the start of each loop

        bestTourLength = Calculate_Tour_Length(optlist + [optlist[0]], dist_matrix)
        bestListSoFar = optlist.copy()

        for i in range(n):
            for j in range(2, n):
                tempOptList = optlist[0:j] + optlist[j:n][::-1]  # Reversing the segment
                tempTourLength = Calculate_Tour_Length(tempOptList + [tempOptList[0]], dist_matrix)

                if tempTourLength + 1e-12 < bestTourLength:
                    bestTourLength = tempTourLength
                    bestListSoFar = tempOptList
                    improvement = True  # An improvement was found

        if improvement:
            optlist = bestListSoFar  # Update the list to the best tour

    # Make sure the final route starts from the given start node (node 0)
    # If the starting node is not at the beginning, rotate the list to bring it to the front
    while optlist[0] != start_node:
        optlist = optlist[1:] + [optlist[0]]

    final_route = optlist + [optlist[0]]  # Add the return to the start node
    total_distance = Calculate_Tour_Length(final_route, dist_matrix)
    t = time.time() - st 

    return total_distance, final_route, t


def Lin_Kernighan_k_Opt(dist_matrix, k=2, start_node=0):
    st = time.time()
    n = len(dist_matrix)
    
    # Generate initial tour starting from start_node
    optlist = [start_node] + [i for i in range(n) if i != start_node]
    improvement = 1

    while improvement > 0:
        bestTourLength = Calculate_Tour_Length(optlist + [optlist[0]], dist_matrix)
        bestListSoFar = optlist.copy()
        improvement = -1
        
        for i in range(n - k + 1):
            for j in range(i + k, n):
                # Extract the subsequence to be flipped
                subsequence = optlist[i:j]
                flipped_subsequence = subsequence[::-1]
                
                # Create a new tour with the flipped subsequence
                tempOptList = optlist[:i] + flipped_subsequence + optlist[j:]
                tempTourLength = Calculate_Tour_Length(tempOptList + [tempOptList[0]], dist_matrix)

                # Check if the new tour is shorter
                if tempTourLength + 1e-12 < bestTourLength:
                    improvement = bestTourLength - tempTourLength
                    bestTourLength = tempTourLength
                    bestListSoFar = tempOptList
        
        if bestTourLength + 1e-12 < Calculate_Tour_Length(optlist + [optlist[0]], dist_matrix):
            optlist = bestListSoFar
        else:
            break

    # Ensure the final route starts from the given start_node (not rotated)
    while optlist[0] != start_node:
        optlist = optlist[1:] + [optlist[0]]

    final_route = optlist + [optlist[0]]  # Add the return to the start node
    total_distance = Calculate_Tour_Length(final_route, dist_matrix)
    t = time.time() - st 

    return total_distance, final_route, t


def build_graph(dist_matrix_):
    n = len(dist_matrix_)
    graph = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i][j] = dist_matrix_[i][j]
    return graph


# prim's method to find minimum spanning tree
import heapq
def prim(graph, start_node=None):
    
    # if start not assigned, use first node in graph
    if start_node is None:
        start_node = next(iter(graph))

    # set to store all visited nodes
    visited_nodes = {start_node}

    # edges for mst
    mst_edges = []

    edge_heap = []
    for neighbor, weight in graph[start_node].items():
        heapq.heappush(edge_heap, (weight, start_node, neighbor))

    # While there are edges to consider and we haven’t covered all nodes:
    while edge_heap and len(visited_nodes) < len(graph):
        edge_weight, from_node, to_node = heapq.heappop(edge_heap)

        # skip if edges matched with visited nodes
        if to_node in visited_nodes:
            continue

        visited_nodes.add(to_node)
        mst_edges.append((from_node, to_node, edge_weight))

        for next_neighbor, next_weight in graph[to_node].items():
            if next_neighbor not in visited_nodes:
                heapq.heappush(edge_heap, (next_weight, to_node, next_neighbor))

    return mst_edges

def find_odd_degree_vertices(mst):
    #  mst is of the form: list(<vertex_1, vertex_2, distance>)
    degree_count = {}
    
    for from_node, to_node, _ in mst:
        degree_count[from_node] = degree_count.get(from_node, 0) + 1
        degree_count[to_node]   = degree_count.get(to_node,   0) + 1

    odd_degree_nodes = []
    for node, degree in degree_count.items():
        if degree % 2 == 1:
            odd_degree_nodes.append(node)

    
    return odd_degree_nodes

def find_minimum_perfect_matching_networkx(dist_matrix_, odd_nodes):

    # form a graph but puting in
    G = nx.Graph()
    for i in range(len(odd_nodes)):
        for j in range(i+1, len(odd_nodes)):
            u, v = odd_nodes[i], odd_nodes[j]
            G.add_edge(u, v, weight=-dist_matrix_[u][v])

    # return a (u, v) pair, which are the minimum-weight perfect matching
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    # add the weight to the tuple, and return
    matching_edges = [(u, v, dist_matrix_[u][v]) for u, v in matching]
    
    return matching_edges

def find_minimum_perfect_matching_approximate(dist_matrix_, odd_nodes):
    unmatched = set(odd_nodes)
    matching_edges = []

    # iterate through all possible matching, time consuming
    while unmatched:
        u = unmatched.pop()
        min_dist = float('inf')
        min_v = None
        for v in unmatched:
            if dist_matrix_[u][v] < min_dist:
                min_dist = dist_matrix_[u][v]
                min_v = v
        unmatched.remove(min_v)
        matching_edges.append((u, min_v, dist_matrix_[u][min_v]))
    
    return matching_edges


def union_mst_and_matching(mst, perfect_matching):
    return mst + perfect_matching

def remove_repeated_vertices(euler_path):
    # remove repeated nodes,
    # e.g.: A->B->A->C becomes A->B->C, second A is removed from our path
    visited = set()
    final_path = []
    
    for v in euler_path:
        if v not in visited:
            visited.add(v)
            final_path.append(v)
            
    if final_path:
        final_path.append(final_path[0])
    return final_path

def find_euler_tour(edge_list):
    """
    multiedges: list of edges in your Eulerian multigraph.
                Each edge can be a 2-tuple (u, v) or a 3-tuple (u, v, weight).
    Returns an Eulerian circuit as a list of vertices.
    """
    # build adjacency list; drop any extra entries after the first two
    adjacency = {}
    for edge in edge_list:
        # edge is of the form (pointA, pointB, weight)
        u = edge[0]
        v = edge[1]
        
        # add v to u’s list
        adjacency.setdefault(u, []).append(v)
        # add u to v’s list
        adjacency.setdefault(v, []).append(u)


    # pick a start vertex that actually has edges
    start_node = next(iter(adjacency))

    # Hierholzer’s algorithm, generate a sequence of path, to go over all nodes
    stack = [start_node]
    path = []

    while stack:
        curr_node = stack[-1]
        if adjacency[curr_node]:
            # still has an unused edge: walk it
            next_node = adjacency[curr_node].pop()
            # remove the back‐edge
            adjacency[next_node].remove(curr_node)
            stack.append(next_node)
        else:
            # no more edges here: record and backtrack
            path.append(stack.pop())


    path.reverse()
    
    return remove_repeated_vertices(path)

def Christoph_Deschauer(dist_matrix_, start_node=None, exact_flag=False):
    import time
    st = time.time()
    
    # Step 1: Build MST
    graph = build_graph(dist_matrix_)
    mst = prim(graph)
    odds = find_odd_degree_vertices(mst)

    # Step 2: Matching odds
    if exact_flag:
        odds_matching = find_minimum_perfect_matching_networkx(dist_matrix_, odds)
    else:
        odds_matching = find_minimum_perfect_matching_approximate(dist_matrix_, odds)
    
    # Step 3: Union MST and matching
    union = union_mst_and_matching(mst, odds_matching)

    # Step 4: Find Eulerian tour
    final_path = find_euler_tour(union)
    min_dist = Calculate_Tour_Length(final_path, dist_matrix_)
    t = time.time() - st

    if start_node is not None:
        if start_node in final_path:
            idx = final_path.index(start_node)
            final_path = final_path[idx:] + final_path[1:idx+1]
    
    return min_dist, final_path, t


