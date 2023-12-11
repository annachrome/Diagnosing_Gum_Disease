# https://colab.research.google.com/drive/1b6X0m_BNH6Vld3hMMJx3TPO3nWXn8enq?authuser=1#scrollTo=_LCoIkarhfYD

import networkx as nx
import matplotlib.pyplot as plt

### 1

G = nx.karate_club_graph()
# print(type(G)) # <class 'networkx.classes.graph.Graph'>

G = nx.Graph(G) # needed for VSCode typing/param hints
# print(type(G)) # <class 'networkx.classes.graph.Graph'>

nx.draw(G, with_labels = True)
#plt.show()

total_deg = 0
for (_, deg) in G.degree:
    total_deg += deg

# print(total_deg/G.number_of_nodes())


def average_degree(num_edges, num_nodes):
    avg_degree = 2*num_edges/num_nodes
    return round(avg_degree)

# print(average_degree(G.number_of_edges(), G.number_of_nodes()))



def avg_clustering_coefficient(G):
    total_cc = sum(nx.clustering(G).values())
    return round(total_cc/G.number_of_nodes(), 2)

# print(avg_clustering_coefficient(G))


def one_iter_pagerank(G, node_id):
    return nx.pagerank(G)[node_id]

# print(one_iter_pagerank(G, 0))


### 3

import torch
import torch.nn as nn

emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
print(f'Sample embedding layer: {emb_sample}') # Embedding(4, 8)

# .LongTensor = .int64
id  = torch.LongTensor([1]) # tensor([1]) 

# Select an embedding in the sample
#print(id, emb_sample(id)) # tensor([[ 0.2834, ..., 0.0071]], grad_fn=<EmbeddingBackward0>)

# Get shape of embedding weight matrix 
shape = emb_sample.weight.data.shape
#print(shape) # torch.Size([4, 8])

# Check if emb is indeed initialized
ids = torch.LongTensor([0, 3])
#print(emb_sample(ids))


# Create node embedding matrix for G - init uniform dist
torch.manual_seed(1)

def create_node_emb(num_node=34, embedding_dim=16):
    '''
    Returns (torch.nn.Embedding layer) node embedding matrix. 
    '''
    emb = nn.Embedding(num_node, embedding_dim)
    
    emb.weight.data.uniform_(0, 1) # _ means operation performed in-place 
    # same: nn.init.uniform_(emb.weight, 0, 1)

    return emb

emb = create_node_emb()
# print(emb.weight.data)

# An example that gets the embeddings for first and last node in G
ids = torch.LongTensor([0, 33])
# print(emb(ids))



### Visualize emb in 2D space with PCA
from sklearn.decomposition import PCA

def visualize_emb(emb):
    X = emb.weight.data.numpy()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    print(components)

    plt.figure(figsize=(6, 6))
    group1_x = []
    group1_y = []
    group2_x = []
    group2_y = []

    # entire node attribute dict (n, ddict) io just nodes n.
    for node in G.nodes(data=True): 
        if node[1]['club'] == 'Mr. Hi':
            group1_x.append(components[node[0]][0])
            group1_y.append(components[node[0]][1])
        else:
            group2_x.append(components[node[0]][0])
            group2_y.append(components[node[0]][1])
    plt.scatter(group1_x, group1_y, color="red", label="Mr. Hi")
    plt.scatter(group2_x, group2_y, color="blue", label="Officer")
    plt.legend()
    plt.show()

# visualize_emb(emb)


### 7: Training the embedding for classifying edges as +ve/-ve
# +ve means edge exists, -ve means edge DNE

from torch.optim import SGD
import torch.nn as nn

pos_edge_list = []
for edge in G.edges():
    pos_edge_list.append(edge)

neg_edge_list = []
K_34_edges = [(a, b) for a in range(34) for b in range(34) if b > a] 

for tuple in K_34_edges:
    if tuple not in pos_edge_list:
        neg_edge_list.append(tuple)

def list_to_tensor(a_list):
    '''
    inputs list of x edges
    returns tensor of shape (2, x)
    '''
    a_tensor = torch.Tensor(a_list).float()
    return a_tensor.t()

pos_edge_index = list_to_tensor(pos_edge_list)
neg_edge_index = list_to_tensor(neg_edge_list)

#print("edge tensors", pos_edge_index, neg_edge_index)


def accuracy(pred, label):
    """
    pred: (tensor) after sigmoid
    label: (torch.LongTensor)

    if pred > 0.5, class as label 1, else label 0

    returns accuracy (float) rounded to 4dp
    """

    pred_labels = (pred > 0.5).float()
    print(pred_labels)

    correct = (pred_labels == label).float().sum()

    accuracy = correct / label.shape[0]
    print(accuracy)

    # item() returns single-value tensor as number
    accuracy = round(accuracy.item(), 4)

    return accuracy


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # (1) Get the embeddings of the nodes in train_edge
    # (2) Dot product the embeddings between each node pair
    # (3) Feed the dot product result into sigmoid
    # (4) Feed the sigmoid output into the loss_fn
    # (5) Print both loss and accuracy of each epoch
    # (6) Update the embeddings using the loss and optimizer
    # (as a sanity check, the loss should decrease during training)
    epochs = 20
    lr = .01
    optimizer = SGD(emb.parameters(), lr=lr, momentum=.9)

    for i in range(epochs):
        pass


# Generate the +ve and -ve labels
pos_labels = torch.ones(pos_edge_index.shape[0])
neg_labels = torch.zeros(neg_edge_index.shape[0])
train_labels = torch.cat([pos_labels, neg_labels], dim=0)
print(f"pos_labels {pos_labels.shape}, neg_labels {neg_labels.shape}")
print("train_labs", train_labels.shape)

# Since the network is very small, we do not split the edges into val/test sets
print(pos_edge_index.shape)
train_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print("train_edges", train_edges.shape)

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
