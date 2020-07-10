# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:04:15 2020

@author: Sadeq Ismailzadeh
"""

#%% Import

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.utils import powerlaw_sequence

num_nodes = 1000
exp = 2.1


sequence=[]
while len(sequence) < num_nodes:
    nextval = int(nx.utils.powerlaw_sequence(1, exp)[0]) #100 nodes, power-law exponent 2.5
    if nextval >= 2 and  nextval <= num_nodes // 2:
        sequence.append(nextval)

print(sequence)

if sum(sequence)%2 == 1:
    sequence[0] += 1
    
graph = nx.configuration_model(sequence)
# count parallel edges and avoid counting A-B as well as B-A
num_edges = sum(len(graph[node][neigh]) for node in graph for neigh in graph.neighbors(node))
print("multigraph has {0:d} edges".format(num_edges))
num_par = sum(len(graph[node][neigh]) for node in graph for neigh in graph.neighbors(node)) // 2
print("multigraph has {0:d} parallel edges".format(num_par))
num_loops = nx.number_of_selfloops(graph)
print("multigraph has {0:d} self-loops".format(num_loops))

graph = nx.Graph(graph)
loops = nx.selfloop_edges(graph)
graph.remove_edges_from(loops)
# get largest connected component
# unfortunately, the iterator over the components is not guaranteed to be sorted by size
largest_cc = max(nx.connected_components(graph), key=len)
graph = graph.subgraph(largest_cc).copy()
                     
print("lcc graph has {0:d} nodes".format(nx.number_of_nodes(graph)))
print("lcc graph has {0:d} edges".format(nx.number_of_edges(graph)))


