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

# to plot SIS simulation results
from bokeh.io import show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

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

plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
plt.plot(degree, np.array(counts) / num_nodes, 'ro')

x = np.arange(1, len(counts))
label = ("$y = x^{{{}}}$").format(-exp)
plt.plot(x, np.power(x, -exp), label=label)
plt.xlabel(r"Degree $k$")
plt.xscale("log")
plt.ylabel(r"$P(k)$")
plt.yscale("log")
plt.legend(loc="best")

# %% Run SIS simulation

# Model selection
model = ep.SISModel(graph)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.2)
cfg.add_model_parameter('lambda', 1)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)
trends = model.build_trends(iterations)

# plot fractions of infected and susceptible nodes in terms of time


viz = DiffusionTrend(model, trends)
p = viz.plot(width=600, height=600)
show(p)

# %% Find the fraction of infected nodes at the stationary state
list_inf =[]
for i in range(-50, 0):
    list_inf.append(iterations[i]["node_count"][0])

list_inf = np.array(list_inf)
list_frac_inf = np.divide(list_inf, num_nodes)

mean_inf_frac = np.mean(list_frac_inf)
variance_inf_frac = np.std(list_frac_inf, axis=0)

print("farction of infected nodes at stationary state is: \n {0:.5f}".format(mean_inf_frac))
print("with standard deviation: \n {0:.5f}".format(variance_inf_frac))

# %%  plot the stationary value of infected nodes in terms of beta

# Model selection

range_beta = np.arange(0.0, 5.0, 0.5)
list_inf_beta = []
list_error =[]

for beta in range_beta:
    model = ep.SISModel(graph)
    # Model Configuration

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)
    cfg.add_model_parameter('lambda', 1)
    cfg.add_model_parameter("fraction_infected", 0.05)
    model.set_initial_status(cfg)

    # Simulation execution
    iterations = model.iteration_bunch(500)

    list_inf = []
    for i in range(-30, 0):
        list_inf.append(iterations[i]["node_count"][0])

    list_inf = np.array(list_inf)
    list_frac_inf = np.divide(list_inf, num_nodes)

    mean_inf_frac = np.mean(list_frac_inf)
    variance_inf_frac = np.std(list_frac_inf, axis=0)

    list_inf_beta.append(mean_inf_frac)
    list_error.append(variance_inf_frac)

plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
plt.errorbar(range_beta, list_inf_beta , yerr=list_error, ecolor="red", capsize=3)
plt.xlabel(r"beta")
plt.ylabel(r"fraction of infected nodes at stationary")