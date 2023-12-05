# PyG for Python 3.8-3.11
# travel-agent-lam env

import networkx as nx

G = nx.Graph()
print(G.is_directed())

H = nx.DiGraph()
print(H.is_directed)

#Add graph level attribute
G.graph["name"] = "bar"
print(G.graph)