import random
import time

import networkx as nx


def main():
    size = 1500
    graph_seed = 'Maui'
    path_seed = 'Ada'
    test(size, graph_seed, path_seed)
    exit()


def get_google_graph():
    file = open('web-Google.txt', 'rb')
    g = nx.read_adjlist(file)

    # we add weights just to make the graph more topologically interesting for project purposes
    add_random_weight(g)

    return g


def get_random_graph(size, seed=None):
    """wrapper for NetworkX Erdos-Renyi random graph"""
    edge_probability = .003
    forest = nx.fast_gnp_random_graph(size, edge_probability, seed)

    # we don't want a bunch of disconnected subgraphs, so we prune before returning
    # nx...subgraphs returns an iterator over graphs that are disconnected in the forest.
    # and we just have to pick the largest subgraph as our own, where nx.number_of_nodes() defines 'largest'
    g = max(nx.connected_component_subgraphs(forest), key=nx.number_of_nodes)

    # we add weights just to make the graph more topologically interesting for project purposes
    add_random_weight(g)

    return g


def get_grid_graph(size):
    g = nx.grid_2d_graph(size, size)
    add_random_weight(g)

    return g


def get_barabasi_graph(size, num_edges, seed=None):
    forest = nx.barabasi_albert_graph(size, num_edges, seed)

    g = max(nx.connected_component_subgraphs(forest), key=nx.number_of_nodes)

    add_random_weight(g)
    nx.set_edge_attributes(g, False, 'visited')
    nx.set_node_attributes(g, False, 'visited')

    return g


def draw(graph: nx.Graph, pos, metrics=None, solution_nodes=()):
    """draws nx Graph objects in preparation for matplotlib plotting"""
    print('Drawing graph... This may take a while')
    start = time.process_time()

    # default types must be immutable, so we need to bookeep metrics and solution nodes
    solution_nodes = list(solution_nodes)
    if metrics is None:
        metrics = {}

    # edges are just tuples of adjacent nodes, and our solution path by definition stores adjacent nodes
    solution_edges = [(solution_nodes[i], solution_nodes[i + 1]) for i in range(len(solution_nodes) - 1)]

    goals = [solution_nodes[0], solution_nodes[-1]]
    goal_labels = {goals[0]: 'S', goals[1]: 'T'}
    weight_labels = nx.get_edge_attributes(graph, 'weight')

    # We draw the whole graph with the default color.
    # Then we redraw the path, goal and visited sets using different attributes.
    nx.draw_networkx_nodes(graph, pos, node_color='k', node_size=.5)
    nx.draw_networkx_edges(graph, pos, edge_color='k', width=.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight_labels, font_size=8)

    # since we've stored 'visited' as a property of nodes, we need to construct a list from that attribute dict
    visited_s_nodelist = [k for (k, v) in nx.get_node_attributes(graph, 's_visited').items() if v]
    visited_t_nodelist = [k for (k, v) in nx.get_node_attributes(graph, 't_visited').items() if v]

    if not visited_t_nodelist:
        # There is nothing in the t_nodelist, which tells us we ran a single A*.
        visited_edgelist = [(a, b) for ((a, b), v) in nx.get_edge_attributes(graph, 's_visited').items() if v]
        nx.draw_networkx_nodes(graph, pos, nodelist=visited_s_nodelist, node_color='b', node_size=.5)
        nx.draw_networkx_edges(graph, pos, edgelist=visited_edgelist, edge_color='b', width=1)
    else:
        visited_both_nodelist = list(set(visited_s_nodelist) & set(visited_t_nodelist))
        nx.draw_networkx_nodes(graph, pos, nodelist=visited_s_nodelist, node_color='b', node_size=.5)
        nx.draw_networkx_nodes(graph, pos, nodelist=visited_t_nodelist, node_color='y', node_size=.5)
        nx.draw_networkx_nodes(graph, pos, nodelist=visited_both_nodelist, node_color='g', node_size=.5)

        visited_s_edgelist = [k for (k, v) in nx.get_edge_attributes(graph, 's_visited').items() if v]
        visited_t_edgelist = [k for (k, v) in nx.get_edge_attributes(graph, 't_visited').items() if v]
        visited_both_edgelist = list(set(visited_s_edgelist) & set(visited_t_edgelist))

        nx.draw_networkx_edges(graph, pos, edgelist=visited_s_edgelist, edge_color='b', width=1)
        nx.draw_networkx_edges(graph, pos, edgelist=visited_t_edgelist, edge_color='y', width=1)
        nx.draw_networkx_edges(graph, pos, edgelist=visited_both_edgelist, edge_color='g', width=1)

    nx.draw_networkx_nodes(graph, pos, nodelist=solution_nodes, node_color='r', node_size=2)
    nx.draw_networkx_edges(graph, pos, edgelist=solution_edges, edge_color='r', width=1)
    nx.draw_networkx_nodes(graph, pos, nodelist=goals, node_color='m', node_size=40)
    nx.draw_networkx_labels(graph, pos, goal_labels, font_size=8)

    metrics['drawing_time:'] = time.process_time() - start


def get_random_node(g):
    return random.choice(list(g.nodes))


def add_random_weight(g):
    """Adds an int weight attribute from [0,10) to every edge in g"""
    random.seed('seed')

    # Spring layouts give longer visual paths to lower weights, and I want to invert that
    max_weight = 10
    for (u, v, w) in g.edges(data=True):
        w['weight'] = random.randint(1, 10)
        w['i_weight'] = max_weight - w['weight'] + 1


def test(size, graph_seed, path_seed):
    pass


if __name__ == '__main__':
    main()
