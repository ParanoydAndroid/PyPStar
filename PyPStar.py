import math
from multiprocessing import Process, Manager
import random

import networkx as nx

import PriorityQueue as pq
import Graph


# TODO: possible use random.getstate(), pickled, then recovered with random.setState to ensure I get repeatable tests

class Search:
    """Class to access traversal methods on a networkX graph.  In particular single and parallel A* searches"""

    # because not all graphs are conducive to the same heuristic, we allow the caller to pass their own on instantiation
    # in the event they don't, we have an anonymous constant function that converts the algo to Dijkstra's
    # Passed in heuristic_func must accept two node parameters and must return a number > 0
    def __init__(self, graph: nx.Graph, source, target, heuristic_func=lambda x, y: 1):

        self.graph = graph
        self.source = source
        self.target = target
        self.h = heuristic_func

        # Do I want these in my object?  It feels sort of wrong
        self.parents = {}

        # map representing g(x) in the standard f(x) = g(x) + h(x) A* cost function.
        self.g = {}

        self.metrics = {}

    def A_star(self):

        open_nodes = pq.PriorityQueue()
        open_nodes.push(self.source, 0)

        parents = self.parents
        parents[self.source] = None

        g = self.g
        g[self.source] = 0

        while not open_nodes.empty():
            current = open_nodes.pop()

            if current == self.target:
                break

            # graph[current] returns an adjacency dictionary associated with the current node.  We then listify that
            # dictionary's keys to get only the IDs of adjacent nodes (as opposed to getting a dictionary of node dicts.
            for node in self.graph[current].keys():
                new_cost = g[current] + self.graph[current][node]['weight']

                # Setting these node attributes is not necessary for the actual search
                # but we use these attributes in draw for later analysis

                nx.set_node_attributes(self.graph, {node: {'visited': True}})
                # self.graph.nodes[node]['visited'] = True
                nx.set_edge_attributes(self.graph, {(node, current): {'visited': True}})
                # self.graph.edges[current][node]['visited'] = True

                # this second condition is required for thread-safing
                # but since we're already rechecking for shorter paths, we no longer need to be concerned
                # if h(x) is monotonic or not.
                if node not in g or new_cost < g[node]:
                    g[node] = new_cost
                    priority = new_cost + self.h(node, self.target)
                    open_nodes.push(node, priority)

                    parents[node] = current

        self.metrics['nodes_explored'] = open_nodes.get_max_count()

        # TODO: refactor this to return a path to the instance.  Can probably use a parents dict. to allow
        # create path to figure out what to do, instead of writing two different methods
        return parents

    def _create_path(self):
        # "if self.parents:
        current = self.target
        path = []

        while current != self.source:
            path.append(current)
            current = self.parents[current]

        path.append(self.source)
        path.reverse()
        # error catch some stuff
        return path

    # noinspection PyPep8Naming
    def _B_star_runner(self, source, target, s_visited, t_visited, mu_list):
        open_nodes = pq.PriorityQueue()
        open_nodes.push(source, 0)

        parents = {source: None}
        s_visited[source] = 0

        while not open_nodes.empty():
            priority, current = open_nodes.pop()

            # note we're accessing a manager proxy to the list, not the proxy directly
            # process safety isn't just a good idea; it's the LAW.
            mu = mu_list[0]
            if priority > mu:
                # then we have found the shortest path
                break

            for node in self.graph[current].keys():
                new_cost = s_visited[current] + self.graph[current][node]['weight']

                # unlike the single version, the parallel version can't write node attributes (safely, at least)
                # but since we're passing a list of visited tiles later anyway, we don't need to.

                # ensures we only add the cost for the shortest known path to the given node
                if node not in s_visited or new_cost < s_visited[node]:
                    s_visited[node] = new_cost
                    parents[node] = current

                    # we only want to add to the frontier if we don't already know the optimal path from
                    # node -> t, which is only the case when our backwards search hasn't touched it yet
                    # this implements the recommendation from Goldberg et al. 2004
                    if node not in t_visited:
                        priority = new_cost + self.h(node, target)
                        open_nodes.push(node, priority)
                        parents[node] = current

                    # if the backwards search *has* touched this node then we need to check if the total path
                    # cost (=s_visited[node] + t_visited[node] is the shortest known path, and update as appropriate
                    elif new_cost + t_visited[node] < mu:
                        mu_list[0] = mu = new_cost + t_visited[node]

        # congrats, you've theoretically found something.  So you need to reconstruct the path now, and
        # maybe stop the other path from running and ruining everything

    # noinspection PyPep8Naming
    def bilateral_A_star(self):
        # using 'with' to ensure shared memory closes after we exit the scope
        with Manager() as mgr:

            # {nodeID: cost} dicts so our two branches can share their path information
            s_visited = mgr.dict()
            t_visited = mgr.dict()

            # manager has to coordinate the shortest expected length of the total path
            # so we have manager accept candidates, check them, and propagate the current with a queue
            mu = math.inf
            mu_list = mgr.list(mu)

            # Even though we're passing the self object, we need to pass source and target in explicitly.
            # This is so that our manager class can reverse them for the backwards search.
            # Otherwise both searches would use the same path
            s_p = Process(target=Search._B_star_runner, args=(self,
                                                              self.source,
                                                              self.target,
                                                              s_visited,
                                                              t_visited,
                                                              mu_list))

            t_p = Process(target=Search._B_star_runner, args=(self,
                                                              self.target,
                                                              self.source,
                                                              t_visited,
                                                              s_visited,
                                                              mu_list))

            s_p.start()
            t_p.start()
            s_p.join()
            t_p.join()

            # do some stuff now that you're done


def test(size, graph_seed, path_seed):
    g = Graph.get_random_graph(size, graph_seed)
    random.seed(path_seed)
    source, target = Graph.get_random_node(g), Graph.get_random_node(g)

    search = Search(g, source, target)
    search.A_star()

    fake_solutions = nx.astar_path(g, source, target)
    real_solutions = search._create_path()

    print('source: ', source)
    print('target: ',  target)
    print('library solution: ', fake_solutions)
    print('My solution: ', real_solutions)
    print('Graph Size:', nx.number_of_nodes(g), 'fake size:', len(fake_solutions), 'real size: ', len(real_solutions))
    print('visited nodes:', search.metrics['nodes_explored'])

    Graph.draw(g, real_solutions)


def main():
    size = 500
    graph_seed = 'Maui1'
    path_seed = 'Ada1'

    test(size, graph_seed, path_seed)


main()
