import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import networkx as nx
import matplotlib.pyplot as plt

# --- Model Definitions ---

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # We will initialize fc1 later based on the output size from conv layers
        self.fc1 = None  
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Dynamically calculate the number of features and initialize fc1 if necessary
        if self.fc1 is None:
            n_features = x.size(1) * x.size(2) * x.size(3)  # (batch_size, channels, height, width)
            self.fc1 = nn.Linear(n_features, 128)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# RNN Model
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)  # Flatten MNIST image to sequence (batch_size, 28, 28)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Transformer Model (Fixed with Embedding Layer)
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(28, 512)  # Project input size 28 to 512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)  # Flatten MNIST to (batch_size, 28, 28)
        x = self.embedding(x)  # Project to (batch_size, 28, 512)
        x = self.transformer_encoder(x)  # Pass through transformer encoder
        x = self.fc(x.mean(dim=1))  # Take the mean and pass through the final FC layer
        return x

# --- Training Functions ---

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("Training complete!")
    return model

# --- Graph Analysis Functions ---

# Convert the neural network architecture to a simple graph
def network_to_graph(model):
    graph = nx.DiGraph()
    layers = list(model.children())
    for i, layer in enumerate(layers[:-1]):
        graph.add_edge(i, i+1, weight=layer.weight.numel() if hasattr(layer, 'weight') else 1)
    return graph

# Function to analyze the graph properties
def analyze_graph(graph, model_name):
    print(f"\n{model_name} Graph Analysis:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    # Degree distribution
    degrees = [graph.degree(n) for n in graph.nodes()]
    plt.hist(degrees, bins=5)
    plt.title(f'Degree Distribution for {model_name}')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

    # Clustering coefficient
    clustering = nx.average_clustering(graph.to_undirected())
    print(f"Clustering coefficient: {clustering}")

    # Path lengths
    if nx.is_connected(graph.to_undirected()):
        path_length = nx.average_shortest_path_length(graph.to_undirected())
        print(f"Average path length: {path_length}")
    else:
        print("Graph is not connected, skipping path length calculation.")

# Visualization function for the graph
def visualize_graph(graph, model_name="Graph Visualization"):
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure and axis
    pos = nx.spring_layout(graph)  # Generate layout
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, ax=ax)
    ax.set_title(f'{model_name} Graph')
    plt.show()

# --- Training and Analysis ---

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Train CNN
cnn = SimpleCNN()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
cnn = train_model(cnn, train_loader, criterion, optimizer_cnn)

# Analyze CNN Graph
cnn_graph = network_to_graph(cnn)
analyze_graph(cnn_graph, "CNN")
visualize_graph(cnn_graph, model_name="CNN")

# Train RNN
rnn = SimpleRNN()
optimizer_rnn = optim.Adam(rnn.parameters(), lr=0.001)
rnn = train_model(rnn, train_loader, criterion, optimizer_rnn)

# Analyze RNN Graph
rnn_graph = network_to_graph(rnn)
analyze_graph(rnn_graph, "RNN")
visualize_graph(rnn_graph, model_name="RNN")

# Train Transformer (fixed)
transformer = SimpleTransformer()
optimizer_transformer = optim.Adam(transformer.parameters(), lr=0.001)
transformer = train_model(transformer, train_loader, criterion, optimizer_transformer)

# Analyze Transformer Graph
transformer_graph = network_to_graph(transformer)
analyze_graph(transformer_graph, "Transformer")
visualize_graph(transformer_graph, model_name="Transformer")
