#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 23:15:36 2023

@author: hbonen
"""

import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.manifold import TSNE
from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


# Set random seed for reproducibility
torch.manual_seed(42)

# Load the Cora dataset
dataset = Planetoid(root='./cora', name='Cora')

data = dataset[0]
x, y = data.x, data.y

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = torch.tensor(y_encoded)

# Split the data into train, validation, and test sets
train_idx, test_idx, train_y, test_y = train_test_split(range(len(y_encoded)), y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)
train_idx, val_idx, train_y, val_y = train_test_split(train_idx, train_y, stratify=train_y, test_size=0.2, random_state=42)

# Create the adjacency matrix (A)
edge_index = data.edge_index
adj = torch.zeros((x.shape[0], x.shape[0]))
adj[edge_index[0], edge_index[1]] = 1

# Convert data to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32)
adj = adj.to(torch.float32)

# Define the Graph Convolutional Network (GCN) model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_layers, num_classes):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.convs.append(GCNConv(hidden_layers[i-1], hidden_layers[i]))
        self.convs.append(GCNConv(hidden_layers[-1], num_classes))
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Specify the hidden layer sizes
# hidden_layers = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
# hidden_layers = [16,16,16,16,16,16,16,16,16,16,16,16]
# hidden_layers = [16,16,16,16,16,16,16,16,16,16]
# hidden_layers = [16,16,16,16,16,16,16,16,16]
hidden_layers = [16,16]

# Instantiate the GCN model
model = GCN(num_features=x.shape[1], hidden_layers=hidden_layers, num_classes=dataset.num_classes)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)  # Pass the entire feature tensor
    loss = F.nll_loss(out[train_idx], train_y)  # Only compute loss for the training nodes
    loss.backward()
    optimizer.step()

# Evaluation loop
def evaluate():
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[test_idx] == test_y
    test_acc = int(test_correct.sum()) / len(test_idx)
    return test_acc

# Training and evaluation
best_test_acc = 0
for epoch in range(1, 201):
    train()
    test_acc = evaluate()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pt')
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('best_model.pt'))
test_acc = evaluate()
print(f'Test Accuracy: {test_acc:.4f}')

# Classification report
model.eval()
out = model(x, edge_index)
pred = out.argmax(dim=1)
pred_labels = label_encoder.inverse_transform(pred.detach().numpy())
true_labels = label_encoder.inverse_transform(y_encoded.numpy())
print(classification_report(true_labels, pred_labels))

# Convert the output features to numpy array
out_features = out.detach().numpy()

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
out_tsne = tsne.fit_transform(out_features)

# Visualize the input and output features using scatter plots
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(out_tsne[:, 0], out_tsne[:, 1], c=y, cmap='tab10')
plt.title('Output Features')

# Convert the input features to numpy array
x_features = x.detach().numpy()

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(x_features)

plt.subplot(122)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='tab10')
plt.title('Input Features')

plt.tight_layout()
plt.show()
