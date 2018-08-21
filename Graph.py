import random
from math import sqrt

import networkx as nx
import matplotlib.pyplot as plot


# size = 15
#
# g = nx.gnp_random_graph(size, .1, 'test')
#
# random.seed('seed')
# for (u, v, w) in g.edges(data=True):
#     w['weight'] = random.randint(0, 10)
#
# nonpath = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= 8]
# path = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > 8]
#
# optimal_spacing = 3/sqrt(size)
#
# pos = nx.spring_layout(g, k=optimal_spacing)
#
# nx.draw_networkx_nodes(g, pos, node_size=300)
# nx.draw_networkx_edges(g, pos, edgelist=path, width=2, edge_color='r')
# nx.draw_networkx_edges(g, pos, edgelist=nonpath, width=2, edge_color='b')
#
# plot.axis('off')
# plot.show()
# print(g.adj)


def get_google_graph():
    file = open('web-Google.txt', 'rb')
    g = nx.read_adjlist(file)

    # we add weights just to make the graph more topologically interesting for project purposes
    random.seed('seed')
    for (u, v, w) in g.edges(data=True):
        w['weight'] = random.randint(0, 10)

    return g


def get_random_graph(size, seed=None):
    """wrapper for NetworkX ERdos-Renyi random graph"""
    edge_probability = .1
    g = nx.fast_gnp_random_graph(size, edge_probability, seed)
    return g


def draw(graph: nx.Graph, solution_nodes=()):
    optimal_spacing = 3 / sqrt(graph.number_of_nodes())

    # default types must be immutable, but we want a list to work on
    solution_nodes = list(solution_nodes)
    solution_edges = [(solution_nodes[i], solution_nodes[i + 1]) for i in range(len(solution_nodes) - 1)]

    # first we determine node spacing using a standard force model
    pos = nx.spring_layout(graph, k=optimal_spacing)

    # we draw the whole graph with the default color
    # then we redraw the path using different attributes
    # this avoids having to partition the graph to draw each only once
    nx.draw_networkx_nodes(graph, pos, node_color='r', node_size=100)
    nx.draw_networkx_edges(graph, pos, edge_color='r', width=1, )

    nx.draw_networkx_nodes(graph, pos, nodelist=solution_nodes, node_color='b', node_size=500)
    nx.draw_networkx_edges(graph, pos, edgelist=solution_edges, edge_color='b', width=2)

    plot.axis('off')
    plot.show()

if __name__ == '__main__':
    g = get_random_graph(30, 'test')
    draw(g)
