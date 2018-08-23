import itertools
from math import sqrt
from multiprocessing import Process, Manager
import random
from os import getpid
import time

import matplotlib.pyplot as plt
import networkx as nx

import PriorityQueue as pq
import Graph


def grid_h(s_node, t_node):
    # Get coords for given nodes
    x0, y0 = s_node
    x1, y1 = t_node

    return abs(x0 - x1) + abs(y0 - y1)


def h_djikstra(x, y):
    return 1


class Search:
    """Class to access traversal methods on a networkX graph.  In particular single and parallel A* searches"""

    # because not all graphs are conducive to the same heuristic, we allow the caller to pass their own on instantiation
    # in the event they don't, we have an anonymous constant function that converts the algo to Dijkstra's
    # Passed in heuristic_func must accept two node parameters and must return a number > 0
    def __init__(self, graph: nx.Graph, source, target, heuristic_func=h_djikstra):

        self.graph = graph
        self.source = source
        self.target = target
        self.h = heuristic_func

        # Used to recreate paths after search is performed.
        # Each stores the search beginning at source/target respectively.
        self.path = []

        self.metrics = {}

    def A_star(self):

        open_nodes = pq.PriorityQueue()
        open_nodes.push(self.source, 0)

        parents = {self.source: None}

        # Map representing g(x) in the standard f(x) = g(x) + h(x) A* cost function.
        g = {self.source: 0}

        while not open_nodes.empty():
            _, current = open_nodes.pop()

            if current == self.target:
                break

            # graph[current] returns an adjacency dictionary associated with the current node.  We then listify that
            # dictionary's keys to get only the IDs of adjacent nodes (as opposed to getting a dictionary of node dicts.
            for node in self.graph[current].keys():
                new_cost = g[current] + self.graph[current][node]['weight']

                # Setting these node attributes is not necessary for the actual search
                # but we use these attributes in draw for later analysis

                nx.set_node_attributes(self.graph, {node: {'s_visited': True}})
                nx.set_edge_attributes(self.graph, {(node, current): {'s_visited': True}})

                if node not in g or new_cost < g[node]:
                    g[node] = new_cost
                    priority = new_cost + self.h(node, self.target)
                    open_nodes.push(node, priority)

                    parents[node] = current

        self.metrics['nodes_explored'] = open_nodes.get_max_count()
        self.path = self._create_path(parents)

        return self.path

    def _create_path(self, parents: dict):
        # "if self.s_parents:
        current = self.target
        path = []

        while current != self.source:
            path.append(current)
            current = parents[current]

        path.append(self.source)
        path.reverse()

        return path

    def _splice_path(self, s_parents, t_parents, s_costs, t_costs):
        """ Recreates a s->t path from a bilateral search returning dicts of s_parents from s->n and n<-t searches"""

        # After we return from this function, we won't know which search found which elements of the path.
        # So we set that information now while we construct the path.

        # I have no idea how to set the edges at this moment.
        s_visiteds = zip(s_costs.keys(), itertools.repeat({'s_visited': True}))
        t_visiteds = zip(t_costs.keys(), itertools.repeat({'t_visited': True}))

        s_visiteds = dict(s_visiteds)
        t_visiteds = dict(t_visiteds)


        nx.set_node_attributes(self.graph, s_visiteds)
        nx.set_node_attributes(self.graph, t_visiteds)

        # There may be many potential paths in our data set.
        # First we find mutually explored nodes
        s_touched = set(s_parents.keys())
        t_touched = set(t_parents.keys())
        mutual_set = s_touched & t_touched
        mutual_costs = []

        # Then we calculate the path lengths through each one of them from both ends
        for node in mutual_set:
            path_length = s_costs[node] + t_costs[node]
            mutual_costs.append((path_length, node))

        min_path, key_tile = min(mutual_costs)
        self.metrics['path_cost'] = min_path
        self.metrics['nodes_explored'] = len(s_costs) + len(t_costs)

        path = []
        current = key_tile
        while current != self.source:
            # I don't feel right about appending and reversing the first half of the list, like it might go wrong ...
            # Instead we push in reverse.
            path.insert(0, current)
            current = s_parents[current]

        path.insert(0, self.source)

        # Begin the last half of the path.  Note we start with the parent, since the key_tile has already been added.
        current = t_parents[key_tile]
        while current != self.target and current is not None:
            path.append(current)
            current = t_parents[current]

        # For small graphs, sometimes one process finds the whole path, so this will end up being redundant
        if path[-1] != self.target:
            path.append(self.target)

        self.metrics['path_length'] = len(path)
        return path

    def _B_star_runner(self, source, target, s_visited_node_costs, t_visited_node_costs, mu_list, resultsq, barrier, metrics):

        # Barrier is a special lock that releases only when all member threads call wait (2, in this case).
        barrier.wait()

        direction = 'x'
        if source == self.source:
            print('process {} started forward search'.format(getpid()))
            direction = 's'
        else:
            print('process {} started reverse search'.format(getpid()))
            direction = 't'

        open_nodes = pq.PriorityQueue()
        open_nodes.push(source, 0)

        parents = {source: None}
        s_visited_node_costs[source] = 0
        visited_edges = {}
        key = '{}_visited'.format(direction)

        while not open_nodes.empty():
            priority, current = open_nodes.pop()

            # note we're accessing a manager proxy to the list, not the proxy directly
            # process safety isn't just a good idea; it's the LAW.
            mu = mu_list[0]

            if priority > mu:

                # If mu has been set to 0 by our partner, we just leave
                if 0 == mu:
                    break

                # We have found the shortest path and we want to break and break our partner as well.
                # First we'll permanently save the right value then set a poison pill.
                mu_list.append(mu)
                mu_list[0] = 0
                break

            for node in self.graph[current].keys():
                new_cost = s_visited_node_costs[current] + self.graph[current][node]['weight']

                # Unlike the single version, the parallel version can't write graph attributes (safely, at least).
                # So, we write to a local dict and then pass to a shared memory object
                visited_edges[(node, current)] = {key: True}

                # ensures we only add the cost for the shortest known path to the given node.
                # The second condition is required for thread-safing.
                # Since we're already rechecking for shorter paths, we no longer need to be concerned
                # if h(x) is monotonic or not.
                if node not in s_visited_node_costs or new_cost < s_visited_node_costs[node]:
                    s_visited_node_costs[node] = new_cost
                    parents[node] = current

                    # we only want to add to the frontier if we don't already know the optimal path from
                    # node -> t, which is only the case when our backwards search hasn't touched it yet
                    # this implements the recommendation from Goldberg et al. 2004
                    if node not in t_visited_node_costs:
                        priority = new_cost + self.h(node, target)
                        open_nodes.push(node, priority)
                        parents[node] = current

                    # if the backwards search *has* touched this node then we need to check if the total path
                    # cost is the shortest known path, and update as appropriate
                    elif new_cost + t_visited_node_costs[node] < mu:
                        mu_list[0] = mu = new_cost + t_visited_node_costs[node]

        resultsq.put(parents)  # blocks
        metrics[direction] = visited_edges

        if source == self.source:
            print('process {} finished forward search'.format(getpid()))
        else:
            print('process {} finished reverse search'.format(getpid()))

        return

    def bilateral_A_star(self):
        # using 'with' to ensure shared memory closes after we exit the scope
        with Manager() as mgr:
            # {nodeID: cost} dicts so our two branches can share their path information
            s_visited_node_costs = mgr.dict()
            t_visited_node_costs = mgr.dict()

            # manager has to coordinate the shortest expected length of the total path
            mu = float('inf')
            mu_list = mgr.list()
            mu_list.append(mu)

            # To gather the two final paths determined to contain the right total path
            resultsq = mgr.Queue()

            # To sync thread start
            barrier = mgr.Barrier(2)

            # Again, not necessary for the code, but used later for report and analysis
            visited_edges = mgr.dict()

            # Even though we're passing the self object, we need to pass source and target in explicitly.
            # This is so that our manager class can reverse them for the backwards search.
            # Otherwise both searches would use the same path
            s_p = Process(target=self._B_star_runner, args=(self.source,
                                                            self.target,
                                                            s_visited_node_costs,
                                                            t_visited_node_costs,
                                                            mu_list,
                                                            resultsq,
                                                            barrier,
                                                            visited_edges))

            t_p = Process(target=self._B_star_runner, args=(self.target,
                                                            self.source,
                                                            t_visited_node_costs,
                                                            s_visited_node_costs,
                                                            mu_list,
                                                            resultsq,
                                                            barrier,
                                                            visited_edges))

            s_p.start()
            t_p.start()
            s_p.join()

            t_p.join()

            parents = []
            while not resultsq.empty():
                parents.append(resultsq.get())  # blocks

            if len(parents) != 2:
                raise ChildProcessError('Received unexpected number of results: {}: {}'.format(len(parents), parents))

            # Now we have to figure out which dict is which
            if self.source in parents[0]:
                s_parents = parents[0]
                t_parents = parents[1]
            else:
                s_parents = parents[1]
                t_parents = parents[0]

            # Now we need to pass all our information into a function to actual get us a path to return
            print('Received all thread results.  Beginning to splice final path.')
            # print('s thread: {}'.format(s_parents))
            # print('t thread: {}'.format(t_parents))
            # print('intersection:', set(s_parents.keys()) & set(t_parents.keys()))
            self.path = self._splice_path(s_parents, t_parents, s_visited_node_costs, t_visited_node_costs)

            nx.set_edge_attributes(self.graph, visited_edges['s'])
            nx.set_edge_attributes(self.graph, visited_edges['t'])

            print('Path successfully spliced')
            return self.path


def test(size, graph_seed, path_seed):
    g = Graph.get_random_graph(size, graph_seed)
    random.seed(path_seed)
    source, target = Graph.get_random_node(g), Graph.get_random_node(g)

    search = Search(g, source, target)
    b_search = Search(g, source, target)

    real_solutions = search.A_star()
    bilat_solutions = b_search.bilateral_A_star()

    print('source: ', source)
    print('target: ', target)

    print('My Single solution: ', real_solutions)
    print('My grade and life at this moment:', bilat_solutions)
    print('Graph Size: {}, single vs double size: {} - {}'.format(nx.number_of_nodes(g),
                                                                  len(real_solutions),
                                                                  len(bilat_solutions)))
    print('Single Metrics: {}'.format(search.metrics))
    print('Bilat Metrics: {}'.format(b_search.metrics))

    # Graph.draw(g, real_solutions)


def test_grid(size, graph_seed, path_seed):
    g = Graph.get_grid_graph(size)
    Graph.add_random_weight(g)
    g2 = Graph.get_random_graph(500, graph_seed)
    g3 = nx.convert_node_labels_to_integers(g)
    random.seed(path_seed)
    source, target = Graph.get_random_node(g), Graph.get_random_node(g)
    search = Search(g, source, target, grid_h)

    g_path = search.A_star()
    b_path = search.bilateral_A_star()
    pos = nx.spring_layout(g, iterations=100, weight='i_weight')
    # nx.draw(g, pos, with_labels=True)
    Graph.draw(g, pos, b_path)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_size=8)
    plt.axis('off')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig("figure{}.png".format(timestr), dpi=1500)
    print('nodes:{}'.format(g.nodes))
    print('g2 nodes: {}'.format(g2.nodes))
    print('g3 nodes: {}'.format(g3.nodes))
    print('path: {}, b_path: {}'.format(g_path, b_path))
    print('lengths {} - {}'.format(len(g_path), len(b_path)))

def main():
    size = 36
    graph_seed = 'Maui1'
    path_seed = 'Ada1'

    # test(size, graph_seed, path_seed)
    test_grid(int(sqrt(size)), graph_seed, path_seed)
    exit(0)


if __name__ == '__main__':
    print("in main:", __name__)
    main()
else:
    print("out of main:", __name__)
