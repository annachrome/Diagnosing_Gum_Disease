# NetworkX 
# travel-agent-lam env
#%%

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
print(G.is_directed())

H = nx.DiGraph()
print(H.is_directed())

# Add graph level attribute
G.graph["name"] = "bar"
print(G.graph)


# Add node_0 with node-level attributes
G.add_node(0,feature=5, label=0)
node_0_attr = G.nodes[0]
print(f"Node 0 has the attributes {node_0_attr}")


# Add multiple nodes with attributes
G.add_nodes_from([
  (1, {"feature": 1, "label": 1}),
  (2, {"feature": 2, "label": 2}),
])

print(G.nodes(data=True)) 
# [(0, {'feature': 5, 'label': 0}), (1, {'feature': 1, 'label': 1}), (2, {'feature': 2, 'label': 2})]

num_nodes = G.number_of_nodes()
print("G has {} nodes".format(num_nodes))

# Add multiple edges with edge weights
G.add_edges_from([
  (1, 2, {"weight": 0.3}),
  (2, 0, {"weight": 0.1}),
  (1, 0, {"weight": 0.5}),
])

nx.draw(G, with_labels=True)

print(f"nodes have degrees {G.degree}")

#%%
J = nx.DiGraph(nx.path_graph(num_nodes))
nx.draw(J, with_labels = True)

pr = nx.pagerank(G, alpha=.8)
print(pr)



plt.show()
#%%

# PyG for Python 3.8-3.11
# PyG = extension lib for Pytorch
# Please setup a virtual environment, e.g.
# Anaconda or Miniconda, or create a Docker image.

# pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
# pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
# pip install -q torch-geometric

import torch
#print(torch.__version__)
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {(2*data.num_edges) / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# %%

print(data.edge_index.T) # edges [a,b]

Data(edge_index=[2, 156], x=[34, 34], y=[34], train_mask=[34])
print(data)

"""
We can see that this data object holds 4 attributes: 
(1) The edge_index property 
(2) node features as x (each of the 34 nodes is assigned a 34-dim feature vector), and to 
(3) node labels as y (each node is assigned to exactly one class). 
(4) train_mask, which describes for which nodes we already know their community assigments. 

In total, we are only aware of the ground-truth labels of 4 nodes (one for each community), 
and the task is to infer the community assignment for the remaining nodes.
"""
# %%

from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)

print(data.x)
print(data.train_mask)


# %%

from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)

        # each layer is 1 hop aggregation
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh() # Final GNN embedding space

        # Apply a final (linear) classifier
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)
print(data.x)


_, h = model(data.x, data.edge_index)
print(f"Embedding shape: {list(h.shape)}")

visualize(h, color=data.y)
# %%

"""
Even before training the weights of our model,
although the weights of our model are initialized completely at random 
Nodes of the same color (community) are already 
closely clustered together in the embedding space.
This leads to the conclusion that GNNs introduce a strong inductive bias, 
leading to similar embeddings for nodes that are close to each other in the input graph.
"""
#%%

import time

model = GCN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.01)



def train(data):
    optimizer.zero_grad() # Clear gradients

    # Call forward method in GCN
    out, h = model(data.x, data.edge_index) # single forward pass
    
    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask],
        ) # Compute the loss solely based on training nodes
    loss.backward() # Derive grads
    optimizer.step() # Update params based on grads

    # Calculate training accuracy on our 4 datapoints
    accuracy = {}
    predicted_classes = torch.argmax(
        out[data.train_mask], 
        axis=1,
        ) # 
    target_classes = data.y[data.train_mask]

    accuracy['train'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float()
    )

    # Calculate val accuracy on whole graph
    predicted_classes = torch.argmax(out, axis=1)
    target_classes = data.y
    accuracy['val'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float()
    )

    return loss, h, accuracy

for epoch in range(300):
    loss, h, accuracy = train(data)

    if epoch % 10 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss, accuracy=accuracy)
        time.sleep(.3)


# %%
