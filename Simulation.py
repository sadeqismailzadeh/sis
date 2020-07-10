# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:04:15 2020

@author: Sadeq Ismailzadeh
"""

#%% Import
import collections
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.utils import powerlaw_sequence

#%% Create Scale-free graph
num_nodes = 1000
exp = 2.2

sequence=[]
while len(sequence) < num_nodes:
    nextval = int(powerlaw_sequence(1, exp)[0])
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

degree_list = [graph.degree(n) for n in graph]

degreeCount = collections.Counter(degree_list)
degree, counts = zip(*degreeCount.items())
#degree = np.log(degree)
#counts = np.log(counts)
plt.plot(degree, np.array(counts)/num_nodes, 'ro')

counts = np.bincount(sequence)
x = np.arange(1, len(counts))

plt.plot(x, np.power(x, -exp))
plt.xlabel(r"Degree $k$")
plt.xscale("log")
plt.ylabel(r"Probability $P(k)$")
plt.yscale("log")
plt.legend(loc="best")

#%% Run SIS simulation
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

# Model selection
model = ep.SISModel(graph)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.1)
cfg.add_model_parameter('lambda', 1)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(50)
trends = model.build_trends(iterations)
#%%
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

viz = DiffusionTrend(model, trends)
p = viz.plot(width=600, height=600)
show(p)