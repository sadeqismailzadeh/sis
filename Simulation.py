# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:04:15 2020

@author: Sadeq Ismailzadeh
"""

# %% Import
import collections
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.utils import powerlaw_sequence

# random number generator
from datetime import datetime

# to simulate sis
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

random.seed(datetime.now())
# %% Create Scale-free graph
num_nodes = 1000
exp = 2.2

sequence = []
while len(sequence) < num_nodes:
    nextval = int(powerlaw_sequence(1, exp)[0])
    if 2 <= nextval <= (num_nodes // 2):
        sequence.append(nextval)

if sum(sequence) % 2 == 1:
    sequence[0] += 1

graph = nx.configuration_model(sequence)

graph = nx.Graph(graph)
loops = nx.selfloop_edges(graph)
graph.remove_edges_from(loops)

# get largest connected component
largest_cc = max(nx.connected_components(graph), key=len)
graph = graph.subgraph(largest_cc).copy()

print("graph has {0:d} nodes and {1:d} edges".format(nx.number_of_nodes(graph),
                                                     nx.number_of_edges(graph)))

degree_list = [graph.degree(n) for n in graph]
degreeCount = collections.Counter(degree_list)
degree, counts = zip(*degreeCount.items())
plt.plot(degree, np.array(counts) / num_nodes, 'ro')

x = np.arange(1, len(counts))

plt.plot(x, np.power(x, -exp), label="f(x) = x^-" + str(exp))
plt.xlabel(r"Degree $k$")
plt.xscale("log")
plt.ylabel(r"Probability $P(k)$")
plt.yscale("log")
plt.legend(loc="best")

# %% Run SIS simulation

# Model selection
model = ep.SISModel(graph)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.5)
cfg.add_model_parameter('lambda', 1)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(500)
trends = model.build_trends(iterations)

# %% plot fractions of infected and susceptible nodes in terms of time
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

viz = DiffusionTrend(model, trends)
p = viz.plot(width=600, height=600)
show(p)

# %% Find the fraction of infected nodes at the stationary state

inf_stationary_list =[]
for i in range(-30, 0):
    inf_stationary_list.append(iterations[i]["node_count"][0])

mean_inf = np.mean(inf_stationary_list)
fraction_inf = np.divide(mean_inf, num_nodes)

print(fraction_inf)
