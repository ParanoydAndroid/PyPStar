import networkx as nx

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

file = open('web-Google.txt', 'rb')
g = nx.read_adjlist(file)
print(g.adj)
print(nx.info(g))

giter = nx.nodes