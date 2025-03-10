"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN

#%%
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCN(data.num_node_features, 16)
        self.conv2 = GCN(16, data.num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) 
    
#%%
class GCN_layer(torch.nn.Module):
    def __init__(self, in_features, out_features, A):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, X):
        return self.fc(torch.spmm(self.A, X)) #이웃 정보 종합

class GCN(torch.nn.Module):
    def __init__(self, num_feature, num_class, A):
        super(GNN, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
                                    GNN_layer(num_feature, 16, A),
                                    nn.ReLU(),
                                    GNN_layer(16, num_class, A)
                                )
        
    def forward(self, X):
        return self.feature_extractor(X)
"""
#%%
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv

class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_classes):
        super(GCN, self).__init__()

        # Initialize the layers
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x