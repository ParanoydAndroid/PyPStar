# Standard libarary
import argparse
import itertools
import random
import sys
import time as t
from math import sqrt
from multiprocessing import Process, Manager
from os import getpid
# Third-party (on pip)
import matplotlib.pyplot as plt
import networkx as nx
# Local
import PriorityQueue as pq
import Graph


Visualize = False


def main(sizes, num_tests, save_graphs):
    global Visualize
    Visualize = save_graphs

    for size in sizes:
        a_results = []
        b_results = []

        for i in range(num_tests):
            graph_seed = None
            path_seed = None

            print('Starting test @ graph size: {}'.format(size))
            a_results.append(test_grid_astar(size, path_seed))
            b_results.append(test_grid_bstar(size, path_seed))

            get_winner(a_results[-1], b_results[-1], size)

            # a_result = test_random_astar(size, graph_seed, path_seed)
            # b_result = test_random_bstar(size, graph_seed, path_seed)
            #
            # get_winner(a_result, b_result, size)
        avg_a = sum(a_results) / len(a_results)
        avg_b = sum(b_results) / len(b_results)

        a_results = ['{:.4f}'.format(f) for f in a_results]
        b_results = ['{:.4f}'.format(f) for f in b_results]

        print("A results: {}".format(a_results))
        print('B results: {}'.format(b_results))
        print('avg A/B at {} nodes: {:.4f}s - {:.4f}s ({:.2f}x)'.format(size, avg_a, avg_b,
                                                                        max(avg_a, avg_b) / min(avg_a, avg_b)))

    exit(0)


# Distance heuristics functions for the Search library. Due to pickling rqs, these must be top level, instead of members
# Any function added here can be called in an argument to Search() to set the heuristic used for that search.
def grid_h(s_node, t_node):
    # Get coords for given nodes
    x0, y0 = s_node
    x1, y1 = t_node

    return abs(x0 - x1) + abs(y0 - y1)


def h_dijkstra(x, y):
    return 1


class Search:
    """Class to access traversal methods on a networkX graph.  In particular single and parallel A* searches"""

    # because not all graphs are conducive to the same heuristic, we allow the caller to pass their own on instantiation
    # in the event they don't, we have an anonymous constant function that converts the algo to Dijkstra's
    # Passed in heuristic_func must accept two node parameters and must return a number > 0
    def __init__(self, graph: nx.Graph, source, target, heuristic_func=h_dijkstra):

        self.graph = graph
        self.source = source
        self.target = target
        self.h = heuristic_func

        # Used to recreate paths after search is performed.
        # Each stores the search beginning at source/target respectively.
        self.path = []

        self.metrics = {}

    def A_star(self):
        start = t.process_time()
        print('Beginning single A* search')

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

        self.metrics['path_cost'] = g[self.target]
        self.metrics['pathfinding_time'] = t.process_time() - start
        self.metrics['nodes_explored'] = len(parents)

        print('Single A* search complete!  Building path ...')
        self.path = self._create_path(parents)
        print('Path sucessfully recreated!')

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
        self.metrics['path_length'] = len(path)

        return path

    def _splice_path(self, s_parents, t_parents, s_costs, t_costs):
        """ Recreates a s->t path from a bilateral search returning dicts of s_parents from s->n and n<-t searches"""

        # After we return from this function, we won't know which search found which elements of the path.
        # So we set that information now while we construct the path.
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
        self.metrics['nodes_explored'] = len(s_touched) + len(t_touched) - len(mutual_set)

        path = []
        current = key_tile
        while current != self.source:
            # I don't feel right about appending and reversing the first half of the list, like it might go wrong ...
            # Instead we push in reverse.
            path.insert(0, current)
            if current in s_parents:
                current = s_parents[current]
            else:
                print('source: {} target: {}'.format(self.source, self.target))
                print(
                    "Sync issue.  source = {}, and current = {}, key tile = {}".format(self.source, current, key_tile))
                print("parents: {}, path: {}".format(s_parents, path))
                break

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

    def _B_star_runner(self, source, target, s_visited_node_costs, t_visited_node_costs, mu_list, resultsq, barrier,
                       metrics):

        # Barrier is a special lock that releases only when all member threads call wait (2, in this case).
        barrier.wait()

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

            # note we're accessing a manager proxy to the list, not the list directly
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

        resultsq.put({direction: parents})  # blocks
        metrics[direction] = visited_edges

        if source == self.source:
            print('process {} finished forward search'.format(getpid()))
        else:
            print('process {} finished reverse search'.format(getpid()))

        return

    def bilateral_A_star(self):
        """Performs two, modified A* searches, each beginning from one end point of this Search object's graph.
          Requires multiprocessing support and outperforms A* for graphs > ~40,000 nodes"""

        # using 'with' to ensure shared memory closes after we exit the scope
        print('setting up bilateral A* search ...')
        start = t.process_time()

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

            # Now we have to figure out which dict is which, so we check the key for its direction indicator
            if 's' in parents[0]:
                s_parents = parents[0]['s']
                t_parents = parents[1]['t']

            elif 't' in parents[0]:
                s_parents = parents[1]['s']
                t_parents = parents[0]['t']
            else:
                raise ChildProcessError('Malformed message, cannot identify owners: {}'.format(parents))

            # Now we need to pass all our information into a function to actual get us a path to return
            print('Received all thread results!  Beginning to splice final path...')
            self.path = self._splice_path(s_parents, t_parents, s_visited_node_costs, t_visited_node_costs)

            nx.set_edge_attributes(self.graph, visited_edges['s'])
            nx.set_edge_attributes(self.graph, visited_edges['t'])

            print('Path successfully spliced!')

            self.metrics['pathfinding_time'] = t.process_time() - start
            return self.path


def test_google_astar(path_seed=None):
    g = Graph.get_google_graph()
    random.seed(path_seed)
    source, target = get_destinations(g)

    search = Search(g, source, target)
    search.A_star()

    m = search.metrics
    m['graph_size'] = nx.number_of_nodes(g)
    m['search_type'] = 'single'

    _visualize(search, nx.number_of_nodes(g))
    return m['pathfinding_time']


def test_google_bstar(path_seed=None):
    g = Graph.get_google_graph()
    random.seed(path_seed)
    source, target = get_destinations(g)

    search = Search(g, source, target)
    search.A_star()

    m = search.metrics
    m['search_type'] = 'bilateral'
    m['graph_size'] = nx.number_of_nodes(g)

    _visualize(search, nx.number_of_nodes(g))
    return m['pathfinding_time']


def test_random_astar(size, graph_seed=None, path_seed=None):
    g = Graph.get_random_graph(size, graph_seed)
    random.seed(path_seed)
    source, target = get_destinations(g)

    search = Search(g, source, target)
    search.A_star()

    m = search.metrics
    m['graph_size'] = size
    m['search_type'] = 'single'

    _visualize(search, size)
    return m['pathfinding_time']


def test_random_bstar(size, graph_seed=None, path_seed=None):
    g = Graph.get_random_graph(size, graph_seed)
    random.seed(path_seed)
    source, target = get_destinations(g)

    search = Search(g, source, target)
    search.bilateral_A_star()

    m = search.metrics
    m['graph_size'] = size
    m['search_type'] = 'bilateral'

    _visualize(search, size)
    return m['pathfinding_time']


def test_grid_astar(size, path_seed=None):
    g = Graph.get_grid_graph(int(sqrt(size)))
    random.seed(path_seed)
    source, target = get_destinations(g)

    search = Search(g, source, target, grid_h)
    search.A_star()

    m = search.metrics
    m['graph_size'] = size
    m['search_type'] = 'single'

    _visualize(search, size)
    return m['pathfinding_time']


def test_grid_bstar(size, path_seed=None):
    g = Graph.get_grid_graph(int(sqrt(size)))

    random.seed(path_seed)
    source, target = get_destinations(g)

    search = Search(g, source, target, grid_h)
    search.bilateral_A_star()

    m = search.metrics
    m['graph_size'] = size
    m['search_type'] = 'bilateral'

    _visualize(search, size)

    return m['pathfinding_time']


def get_destinations(g):
    source, target = Graph.get_random_node(g), Graph.get_random_node(g)

    while source == target:
        source, target = Graph.get_random_node(g), Graph.get_random_node(g)

    return source, target


def get_winner(a_result, b_result, size):
    if a_result < b_result:
        winner = 'single search'
        winner_result = a_result
        loser_result = b_result
    else:
        winner = 'bilateral search'
        winner_result = b_result
        loser_result = a_result

    print('RESULTS: {} wins, with {:.4f}s execution vs {:.4f}s against {} nodes ({:.2f}x)'.format(winner,
                                                                                                  winner_result,
                                                                                                  loser_result,
                                                                                                  size,
                                                                                                  loser_result / winner_result))


def plot(metrics):
    print('Graph drawn!  Saving drawn graph to working directory...')
    m = metrics

    plt.axis('off')
    plt.title('{}-node Graph with {} search'.format(m['graph_size'], m['search_type']))
    bottom = -1.2  # -1.3
    left = -1.2  # -1.25
    right = 1.2  # .7
    offset = .075

    metrics = []
    metrics.insert(0, 'Pathfinding time: {:.2f}s'.format(m['pathfinding_time']))
    metrics.insert(0, 'path length: {}, path cost: {}'.format(m['path_length'], m['path_cost']))
    metrics.insert(0, 'Graph: {} nodes, of which {} were visited'.format(m['graph_size'], m['nodes_explored']))

    for i in range(len(metrics)):
        plt.text(left, bottom + i * offset, metrics[i],
                 fontsize=8, horizontalalignment='left', verticalalignment='bottom')

    legend = []
    legend.insert(0, 'blue: forward searched')
    legend.insert(0, 'yellow: reverse searched')
    legend.insert(0, 'green: both')
    legend.insert(0, 'Red: final path')

    for i in range(len(legend)):
        plt.text(right, bottom + i * offset, legend[i],
                 fontsize=6, horizontalalignment='right', verticalalignment='bottom')

    timestr = t.strftime("%Y%m%d-%H%M%S")
    plt.savefig("figure{}.png".format(timestr), dpi=1500)
    print('Save complete!')
    plt.close('all')


def _visualize(search, size):
    m = search.metrics

    if size < 50000 and Visualize:
        print('Drawing graph... This may take a while')
        g = search.graph
        pos = nx.spring_layout(g, iterations=100, weight='i_weight')
        path = search.path

        Graph.draw(g, pos, m, path)
        # Utility to save .png to working dir
        plot(m)
    else:
        print("Large search finished!")
        print('Pathfinding time: {:.4f}'.format(m['pathfinding_time']))
        print('path length: {}, path cost: {}'.format(m['path_length'], m['path_cost']))
        print('Graph: {} nodes, of which {} were visited'.format(m['graph_size'], m['nodes_explored']))


if __name__ == '__main__':
    sizes_default = [50, 100, 1000, 10000]
    numTests_default = 5

    # Capture CLI arguments.  See PyPstar -h for help.
    parser = argparse.ArgumentParser(description='Create and solve a variety of graphs using A* variants')
    parser.add_argument('-v', '--visualize', action='store_true', dest='v', help='Create .png visualizations in '
                                                                                 'working directory', required=False)
    parser.add_argument('-s', '--sizes', action='store', dest='s', nargs='+', type=int, default=sizes_default,
                        help='Requires space-delimited int arguments. Determines the size range over which to run'
                             ' <numTests={}> tests.\nExample: "PyPStar -s 50, 100"'.format(numTests_default))
    parser.add_argument('-n', '--numTests', action="store", dest="n", type=int, default=numTests_default,
                        help='Requires int argument. Determines the number of tests run at each '
                             '<size={}>.  Example: "PyPStar -n 10"'.format(sizes_default))

    args = parser.parse_args(sys.argv[1:])
    main(args.s, args.n, args.v)
