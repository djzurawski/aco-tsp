from math import sqrt
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

#Note: np.append()'s' potentially 10% slower than normal list append

#Global variables. Still not sure what these should be
SCENT_MIN = 0.000001
ALPHA = 0
BETA = 1
EVAP_COEFF = 0.000005

class Ant(object):
    def __init__(self, node):
        self.node = node
        self.path = np.array([node])

class Node(object):
    def __init__(self, id, x, y):
        self.id = int(id) - 1                               #added -1
        self.x = float(x)
        self.y = float(y)

def build_graph(filename):

    with open (filename, "r") as myfile:
        data = myfile.readlines()

    node_list = []
    length = len(data)
    print "Number of cities = %d" % length
    for i in xrange(length):
        line = data[i].split()
        node = Node(line[0], line[2], line[1])
        node_list.append(node)

    distance = np.empty([length, length], dtype=float)
    heuristic = np.empty([length, length], dtype=float)
    scent = np.empty([length, length], dtype=float)
    scent.fill(SCENT_MIN)

    (distance, heuristic) = init_distances(node_list, distance)

    return (node_list, distance, heuristic, scent)

def init_distances(node_list, distances):

    for node1 in node_list:
        for node2 in node_list:
            dist = sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
            distances[node1.id][node2.id] = dist                    #removed -1's
    np.fill_diagonal(distances, np.nan)

    heuristic = 1/np.sqrt(distances)
    return (distances, heuristic)

#i think something might be wrong with updating probability
def update_probability(scent, heuristic):

    probability = ((scent**ALPHA) * (heuristic**BETA)) / \
            np.nansum((scent**ALPHA) * (heuristic**BETA), axis=1)[:,np.newaxis]

    #pure distance probability
    #probability = (heuristic**BETA) / np.nansum((heuristic**BETA), axis=1)[:,np.newaxis]

    return probability

def update_scents(ants, scent):

    num_nodes = len(ants[0].path)

    for ant in ants:
        dist = sum(ant.path)
        for i in xrange(num_nodes - 1):
            src = ant.path[i]
            dst = ant.path[i + 1]
            #this calc done 2x. Room for improvement
            scent[src][dst] = max((1 - EVAP_COEFF) * scent[src][dst] + 1/dist, SCENT_MIN)
            scent[dst][src] = max((1 - EVAP_COEFF) * scent[dst][src] + 1/dist, SCENT_MIN)

    return scent

def calc_path_length(ant):

    dist = 0
    for i in xrange(ant.path.shape[0]):
        src = ant.path[i]
        dst = ant.path[i + 1]
        dist = dist + distance[src][dst]

    return dist

#I think something is messed up in the inner loop with the offset between nodeID and indexes
def update_paths(ants, probability):
    num_nodes = probability.shape[0]
    ant_id = 0
    for ant in ants:
        new_arr = np.copy(probability)
        curr_node = ant.path[-1]
        for i in xrange(num_nodes):
            for node in ant.path:
                new_arr[curr_node][node] = 0

            row_sum = np.nansum(new_arr[curr_node])
            if row_sum != 0:
                new_arr[curr_node] = new_arr[curr_node] / row_sum
                next_node = np.random.choice(np.arange(0, num_nodes), p = new_arr[curr_node])
                ant.path = np.append(ant.path, next_node)
            else:
                ant.path = np.append(ant.path, ant.path[0])

        ant_id = ant_id + 1

    return ants

def plot(nodes, path):
    locations = {}
    for node in nodes:
        locations[node.id] = (node.x, node.y)

    x = []
    y = []
    nums = []
    for i in path:
        x.append(locations[i][0])
        y.append(locations[i][1])

    plt.scatter(x,y)
    plt.plot(x,y)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Shortest path found")
    plt.show()

def main():

    #parse arguments
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-a', action='store',dest='num_ants',default=10, type=int)
    parser.add_argument('-i', action='store',dest='iterations',default=10, type=int)
    parser.add_argument('-f', action='store',dest='file',type=str)
    args = parser.parse_args()


    #Intialize graph
    num_ants = args.num_ants
    iterations = args.iterations
    file_path = args.file

    min_dist = float('inf')
    (nodes, distance, heuristic, scent) = build_graph(file_path)

    #randomly place ants
    ants = np.empty(num_ants, dtype=object)
    for i in xrange(num_ants):
        ants[i] = Ant(np.random.random_integers(0, distance.shape[0] - 1))

    #main loop
    min_path = []
    for a in xrange(iterations):

        probability = update_probability(scent, heuristic)
        ants = update_paths(ants, probability)

        #find shortest path from ant paths
        for ant in ants:
            dist = 0
            for i in xrange(ant.path.shape[0] - 1):
                src = ant.path[i]
                dst = ant.path[i + 1]
                dist = dist + distance[src][dst]
            if dist < min_dist:
                min_path = ant.path
                min_dist = dist
                print "New minimum distance = %f" % dist

        scent = update_scents(ants, scent)

        #reset ant paths
        for ant in ants:
            ant.path = [ant.path[0]]

    print "Shortest path found:"
    print min_path
    print "Minimum distance found = %f" % min_dist

    plot(nodes,min_path)

if __name__ == '__main__':
    main()