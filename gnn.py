from torch_geometric.nn import GCNConv
import torch_geometric

class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

# Example data structure for PyTorch Geometric
from torch_geometric.data import Data

# Dummy node features and edge index
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)  # Example features
edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edges

data = Data(x=x, edge_index=edge_index)

# Instantiate the GNN
gnn_model = GNN(input_size=1, hidden_size=16, output_size=2)
print(gnn_model)
