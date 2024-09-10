import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt

# Define a simple neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network
model = SimpleNN()

# Visualize the network as a graph
def visualize_network_as_graph(model):
    G = nx.Graph()
    
    # Add nodes and edges for the layers
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_matrix = param.data
            layer_nodes = weight_matrix.shape[0]
            input_nodes = weight_matrix.shape[1]
            G.add_nodes_from(range(layer_nodes + input_nodes))  # Add nodes
            
            # Add edges (assumes dense connections between nodes)
            for i in range(layer_nodes):
                for j in range(input_nodes):
                    if weight_matrix[i, j].abs() > 0.01:  # Threshold to include an edge
                        G.add_edge(i, j + layer_nodes)
    
    return G

# Convert the neural network to a graph and plot
graph = visualize_network_as_graph(model)
nx.draw(graph, with_labels=True)
plt.show()

# Calculate graph properties
def calculate_graph_properties(G):
    print("Node degree distribution:", dict(G.degree()))
    print("Clustering coefficient:", nx.clustering(G))
    print("Shortest path lengths:", dict(nx.shortest_path_length(G)))
    print("Graph density:", nx.density(G))

calculate_graph_properties(graph)

# Function to simulate lobotomy by removing a subset of nodes and checking connectivity
def lobotomy_simulation(G, node_list):
    G_removed = G.copy()
    G_removed.remove_nodes_from(node_list)
    print("After lobotomy, number of connected components:", nx.number_connected_components(G_removed))
    return G_removed

# Example: removing 10 nodes (simulating lobotomy)
lobotomy_simulation(graph, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
