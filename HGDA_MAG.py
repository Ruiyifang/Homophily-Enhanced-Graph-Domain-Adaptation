import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, degree
from pygda.datasets import MAGDataset
from sklearn.metrics import f1_score
import scipy.sparse as sp
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, degree
    from pygda.datasets import MAGDataset
except ModuleNotFoundError as e:
    raise ImportError("Please ensure all necessary libraries such as torch, pygda, and torch_geometric are installed.") from e

# Load datasets
MAG_CN_dataset = MAGDataset('C:/Users/Administrator/Desktop/Graph DA/MAG_US', name='source')[0]
MAG_RU_dataset = MAGDataset('C:/Users/Administrator/Desktop/Graph DA/MAG_JP', name='source')[0]

source_dataset = MAG_CN_dataset
target_dataset = MAG_RU_dataset

# Validate datasets
if not hasattr(source_dataset, 'x') or not hasattr(source_dataset, 'y'):
    raise ValueError("Source dataset is missing features or labels.")
if not hasattr(target_dataset, 'x') or not hasattr(target_dataset, 'y'):
    raise ValueError("Target dataset is missing features or labels.")

if source_dataset.x is None or source_dataset.y is None:
    raise ValueError("Source dataset features or labels are None.")
if target_dataset.x is None or target_dataset.y is None:
    raise ValueError("Target dataset features or labels are None.")


def compute_normalized_adjacency(edge_index, num_nodes):
    """Compute normalized adjacency matrix Ã = D^(-1/2) A D^(-1/2)"""
    # Add self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # Compute degree
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Create normalized adjacency matrix
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    # Convert to sparse tensor
    adj = torch.sparse.FloatTensor(edge_index, edge_weight, (num_nodes, num_nodes))
    
    return adj


def compute_normalized_laplacian(edge_index, num_nodes):
    """Compute normalized Laplacian L̃ = I - D^(-1/2) A D^(-1/2)"""
    adj = compute_normalized_adjacency(edge_index, num_nodes)
    identity = torch.sparse.FloatTensor(
        torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]),
        torch.ones(num_nodes),
        (num_nodes, num_nodes)
    )
    # L̃ = I - Ã
    laplacian = identity - adj
    return laplacian


class SpectralFilter(nn.Module):
    """Base class for spectral filters on graphs"""
    def __init__(self, input_dim, hidden_dim):
        super(SpectralFilter, self).__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, filter_matrix):
        # Apply filter: H^l = ReLU(α * filter_matrix @ H^(l-1) @ W^(l-1))
        filtered = torch.sparse.mm(filter_matrix, x) if filter_matrix.is_sparse else filter_matrix @ x
        transformed = self.weight(filtered)
        output = F.relu(self.alpha * transformed)
        return output


class HomophilicFilter(SpectralFilter):
    """Homophilic (Low-pass) Filter: Uses normalized adjacency Ã"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        
    def forward(self, x, adj_normalized):
        return super().forward(x, adj_normalized)


class FullPassFilter(SpectralFilter):
    """Full-pass Filter: H_F = I (identity matrix)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        
    def forward(self, x, num_nodes):
        # For full-pass, we just use identity (no graph filtering)
        transformed = self.weight(x)
        output = F.relu(self.alpha * transformed)
        return output


class HeterophilicFilter(SpectralFilter):
    """Heterophilic (High-pass) Filter: Uses normalized Laplacian L̃"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        
    def forward(self, x, laplacian_normalized):
        return super().forward(x, laplacian_normalized)


class DomainAlignmentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(DomainAlignmentModel, self).__init__()
        
        self.num_layers = num_layers
        
        # Initialize filters for each layer
        self.homophilic_filters = nn.ModuleList()
        self.fullpass_filters = nn.ModuleList()
        self.heterophilic_filters = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.homophilic_filters.append(HomophilicFilter(in_dim, hidden_dim))
            self.fullpass_filters.append(FullPassFilter(in_dim, hidden_dim))
            self.heterophilic_filters.append(HeterophilicFilter(in_dim, hidden_dim))
        
        # Learnable combination weights for filters
        self.weight_homo = nn.Parameter(torch.tensor(1.0))
        self.weight_full = nn.Parameter(torch.tensor(1.0))
        self.weight_hetero = nn.Parameter(torch.tensor(1.0))
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, num_nodes):
        # Precompute graph matrices
        adj_normalized = compute_normalized_adjacency(edge_index, num_nodes)
        laplacian_normalized = compute_normalized_laplacian(edge_index, num_nodes)
        
        # Process through layers
        h_homo = x
        h_full = x
        h_hetero = x
        
        for i in range(self.num_layers):
            h_homo = self.homophilic_filters[i](h_homo, adj_normalized)
            h_full = self.fullpass_filters[i](h_full, num_nodes)
            h_hetero = self.heterophilic_filters[i](h_hetero, laplacian_normalized)
        
        # Combine filter outputs with learnable weights
        combined = (self.weight_homo * h_homo + 
                   self.weight_full * h_full + 
                   self.weight_hetero * h_hetero)
        
        return combined, h_homo, h_full, h_hetero


# Initialize model
input_dim = source_dataset.x.size(1)
hidden_dim = 64
output_dim = int(source_dataset.y.max().item()) + 1
model = DomainAlignmentModel(input_dim, hidden_dim, output_dim, num_layers=2)

# Optimizer and loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
classification_loss_fn = nn.CrossEntropyLoss()
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

# Prepare data
source_features, source_labels = source_dataset.x, source_dataset.y
source_edge_index = source_dataset.edge_index
source_num_nodes = source_features.size(0)

target_features, target_labels = target_dataset.x, target_dataset.y
target_edge_index = target_dataset.edge_index
target_num_nodes = target_features.size(0)


def train(model, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Source domain forward pass
        src_combined, src_homo, src_full, src_hetero = model(
            source_features, source_edge_index, source_num_nodes
        )
        src_logits = model.classifier(src_combined)
        source_class_loss = classification_loss_fn(src_logits, source_labels)
        
        # Target domain forward pass
        tgt_combined, tgt_homo, tgt_full, tgt_hetero = model(
            target_features, target_edge_index, target_num_nodes
        )
        tgt_logits = model.classifier(tgt_combined)
        
        # Generate pseudo-labels for target domain
        with torch.no_grad():
            pseudo_target_labels = tgt_logits.argmax(dim=1)
        target_class_loss = classification_loss_fn(tgt_logits, pseudo_target_labels)
        
        # Domain alignment loss (KL divergence between filter outputs)
        # Sample nodes for alignment if sizes don't match
        min_nodes = min(src_homo.size(0), tgt_homo.size(0))
        src_indices = torch.randperm(src_homo.size(0))[:min_nodes]
        tgt_indices = torch.randperm(tgt_homo.size(0))[:min_nodes]
        
        kl_homo = kl_loss_fn(
            F.log_softmax(tgt_homo[tgt_indices], dim=1), 
            F.softmax(src_homo[src_indices], dim=1)
        )
        kl_full = kl_loss_fn(
            F.log_softmax(tgt_full[tgt_indices], dim=1), 
            F.softmax(src_full[src_indices], dim=1)
        )
        kl_hetero = kl_loss_fn(
            F.log_softmax(tgt_hetero[tgt_indices], dim=1), 
            F.softmax(src_hetero[src_indices], dim=1)
        )
        kl_loss = kl_homo + kl_full + kl_hetero
        
        # Total loss
        loss = source_class_loss + 0.1 * kl_loss + 0.1 * target_class_loss
        loss.backward()
        optimizer.step()
        
        # Evaluate on target domain
        with torch.no_grad():
            model.eval()
            tgt_combined_eval, _, _, _ = model(
                target_features, target_edge_index, target_num_nodes
            )
            tgt_logits_eval = model.classifier(tgt_combined_eval)
            predictions = tgt_logits_eval.argmax(dim=1)
            test_accuracy = (predictions == target_labels).float().mean().item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, "
                  f"Source Loss: {source_class_loss:.4f}, "
                  f"Target Pseudo Loss: {target_class_loss:.4f}, "
                  f"KL Loss: {kl_loss:.4f}, "
                  f"Target Test Accuracy: {test_accuracy:.4f}")


def test(model):
    model.eval()
    with torch.no_grad():
        tgt_combined, _, _, _ = model(
            target_features, target_edge_index, target_num_nodes
        )
        tgt_logits = model.classifier(tgt_combined)
        predictions = tgt_logits.argmax(dim=1)
        accuracy = (predictions == target_labels).float().mean().item()
        
        true_labels = target_labels.cpu().numpy()
        pred_labels = predictions.cpu().numpy()
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        
        print(f"\n=== Final Results ===")
        print(f"Target Domain Test Accuracy: {accuracy:.4f}")
        print(f"Target Domain Micro-F1: {micro_f1:.4f}")
        print(f"Target Domain Macro-F1: {macro_f1:.4f}")
        
        # Print filter weights
        print(f"\nLearned Filter Weights:")
        print(f"  Homophilic (Low-pass): {model.weight_homo.item():.4f}")
        print(f"  Full-pass: {model.weight_full.item():.4f}")
        print(f"  Heterophilic (High-pass): {model.weight_hetero.item():.4f}")


# Train and test the model
print("Starting training...")
train(model, optimizer, num_epochs=50)
test(model)
