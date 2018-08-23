from multiprocessing import Process, Manager
import random
from os import getpid

import networkx as nx

import PriorityQueue as pq
import Graph


# TODO: Update to use correct h function only.
def h_test_out(x, y):
    return 1


class Search:
    """Class to access traversal methods on a networkX graph.  In particular single and parallel A* searches"""

    def h_test_in(self, x, y):
        return 1

    # because not all graphs are conducive to the same heuristic, we allow the caller to pass their own on instantiation
    # in the event they don't, we have an anonymous constant function that converts the algo to Dijkstra's
    # Passed in heuristic_func must accept two node parameters and must return a number > 0
    def __init__(self, graph: nx.Graph, source, target, heuristic_func=h_test_out):

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
        print('Key tile, min_path:', key_tile, ",", min_path)

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
        while current != self.target:
            path.append(current)
            current = t_parents[current]

        path.append(self.target)

        self.metrics['path_length'] = len(path)
        return path

    def _B_star_runner(self, source, target, s_visited, t_visited, mu_list, resultsq, barrier):
        # Install process barrier from Manager here

        barrier.wait()
        print('pid {} passed barrier'.format(getpid()))
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

                # If mu has been set to 0 by our partner, we just leave
                if 0 == mu:
                    break

                # We have found the shortest path and we want to break and break our partner as well.
                # First we'll permanently save the right value then set a poison pill.
                mu_list.append(mu)
                mu_list[0] = 0
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

        resultsq.put(parents)  # blocks
        return

    def bilateral_A_star(self):
        # using 'with' to ensure shared memory closes after we exit the scope
        with Manager() as mgr:
            # {nodeID: cost} dicts so our two branches can share their path information
            s_visited = mgr.dict()
            t_visited = mgr.dict()

            # manager has to coordinate the shortest expected length of the total path
            mu = float('inf')
            mu_list = mgr.list()
            mu_list.append(mu)

            # To gather the two final paths determined to contain the right total path
            resultsq = mgr.Queue()

            # To sync thread start
            barrier = mgr.Barrier(2)

            # Even though we're passing the self object, we need to pass source and target in explicitly.
            # This is so that our manager class can reverse them for the backwards search.
            # Otherwise both searches would use the same path
            s_p = Process(target=self._B_star_runner, args=(self.source,
                                                            self.target,
                                                            s_visited,
                                                            t_visited,
                                                            mu_list,
                                                            resultsq,
                                                            barrier))

            t_p = Process(target=self._B_star_runner, args=(self.target,
                                                            self.source,
                                                            t_visited,
                                                            s_visited,
                                                            mu_list,
                                                            resultsq,
                                                            barrier))

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
            print('received all thread results')
            print('s thread: {}'.format(s_parents))
            print('t thread: {}'.format(t_parents))
            print('intersection:', set(s_parents.keys()) & set(t_parents.keys()))
            self.path = self._splice_path(s_parents, t_parents, s_visited, t_visited)

            return self.path


def test(size, graph_seed, path_seed):
    g = Graph.get_random_graph(size, graph_seed)
    random.seed(path_seed)
    source, target = Graph.get_random_node(g), Graph.get_random_node(g)

    search = Search(g, source, target)
    b_search = Search(g, source, target)

    # fake_solutions = nx.astar_path(g, source, target)
    real_solutions = search.A_star()
    bilat_solutions = b_search.bilateral_A_star()

    print('source: ', source)
    print('target: ', target)
    # print('library solution: ', fake_solutions)
    print('My Single solution: ', real_solutions)
    print('My grade and life at this moment:', bilat_solutions)
    print('Graph Size: {}, single vs double size: {} - {}'.format(nx.number_of_nodes(g),
                                                                  len(real_solutions),
                                                                  len(bilat_solutions)))
    print('Single Metrics: {}'.format(search.metrics))
    print('Bilat Metrics: {}'.format(b_search.metrics))

    # Graph.draw(g, real_solutions)


def main():
    size = 3000
    graph_seed = 'Maui1'
    path_seed = 'Ada1'

    test(size, graph_seed, path_seed)
    exit(0)


if __name__ == '__main__':
    print("in main:", __name__)
    main()
else:
    print("out of main:", __name__)