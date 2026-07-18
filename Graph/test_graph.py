import networkx as nx

G = nx.read_graphml("Graph/Graphs/10489.graphml")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())