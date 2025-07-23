import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix
from pygda.datasets import AirportDataset
from pygda.datasets import MAGDataset

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.utils import to_scipy_sparse_matrix
    from pygda.datasets import AirportDataset
except ModuleNotFoundError as e:
    raise ImportError("Please ensure all necessary libraries such as torch, pygda, and torch_geometric are installed.") from e

MAG_CN_dataset = MAGDataset('C:/Users/Administrator/Desktop/Graph DA/MAG_US', name='source')[0]
MAG_RU_dataset = MAGDataset('C:/Users/Administrator/Desktop/Graph DA/MAG_JP', name='source')[0]


source_dataset = MAG_CN_dataset
target_dataset = MAG_RU_dataset


if not hasattr(source_dataset, 'x') or not hasattr(source_dataset, 'y'):
    raise ValueError("Source dataset is missing features or labels.")
if not hasattr(target_dataset, 'x') or not hasattr(target_dataset, 'y'):
    raise ValueError("Target dataset is missing features or labels.")

if source_dataset.x is None or source_dataset.y is None:
    raise ValueError("Source dataset features or labels are None.")
if target_dataset.x is None or target_dataset.y is None:
    raise ValueError("Target dataset features or labels are None.")


class DomainAlignmentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DomainAlignmentModel, self).__init__()
        

        self.low_pass = nn.Linear(input_dim, hidden_dim)
        self.high_pass = nn.Linear(input_dim, hidden_dim)
        self.identity = nn.Linear(input_dim, hidden_dim)
        

        self.weight_low = nn.Parameter(torch.tensor(1.0))
        self.weight_high = nn.Parameter(torch.tensor(1.0))
        self.weight_id = nn.Parameter(torch.tensor(1.0))
        

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        low_pass = F.relu(self.low_pass(x))
        high_pass = F.relu(self.high_pass(x))
        identity = F.relu(self.identity(x))
        
        combined = self.weight_low * low_pass + self.weight_high * high_pass + self.weight_id * identity
        return combined, low_pass, high_pass, identity


input_dim = source_dataset.x.size(1)
hidden_dim = 64
output_dim = int(source_dataset.y.max().item()) + 1
model = DomainAlignmentModel(input_dim, hidden_dim, output_dim)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
classification_loss_fn = nn.CrossEntropyLoss()
kl_loss_fn = nn.KLDivLoss(reduction='batchmean')


source_features, source_labels = source_dataset.x, source_dataset.y
source_data = torch.utils.data.TensorDataset(source_features, source_labels)
source_loader = torch.utils.data.DataLoader(source_data, batch_size=32, shuffle=True)

target_features, target_labels = target_dataset.x, target_dataset.y
target_data = torch.utils.data.TensorDataset(target_features, target_labels)
target_loader = torch.utils.data.DataLoader(target_data, batch_size=32, shuffle=True)


def train(model, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_source_loss = 0
        total_target_loss = 0
        total_kl_loss = 0

        for (src_batch, tgt_batch) in zip(source_loader, target_loader):
            optimizer.zero_grad()

            # Source Domain
            src_features, src_labels = src_batch
            src_combined, src_low, src_high, src_id = model(src_features)
            src_logits = model.classifier(src_combined)
            source_class_loss = classification_loss_fn(src_logits, src_labels)

            # Target Domain (pseudo-labels generation and loss computation)
            tgt_features, _ = tgt_batch
            tgt_combined, tgt_low, tgt_high, tgt_id = model(tgt_features)
            tgt_logits = model.classifier(tgt_combined)
            pseudo_target_labels = tgt_logits.argmax(dim=1)  # Pseudo labels from current model predictions
            target_class_loss = classification_loss_fn(tgt_logits, pseudo_target_labels)


            if src_low.size(0) == tgt_low.size(0):  # Ensure batch sizes match for KL alignment
                kl_low = kl_loss_fn(F.log_softmax(tgt_low, dim=1), F.softmax(src_low, dim=1))
                kl_high = kl_loss_fn(F.log_softmax(tgt_high, dim=1), F.softmax(src_high, dim=1))
                kl_id = kl_loss_fn(F.log_softmax(tgt_id, dim=1), F.softmax(src_id, dim=1))
                kl_loss = kl_low + kl_high + kl_id
            else:
                kl_loss = torch.tensor(0.0, requires_grad=True)  # Handle mismatch gracefully


            loss = source_class_loss  + kl_loss +target_class_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_source_loss += source_class_loss.item()
            total_target_loss += target_class_loss.item()
            total_kl_loss += kl_loss.item()


        tgt_combined_eval, _, _, _ = model(target_features)
        tgt_logits_eval = model.classifier(tgt_combined_eval)
        predictions = tgt_logits_eval.argmax(dim=1)
        test_accuracy = (predictions == target_labels).float().mean().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Source Loss: {total_source_loss:.4f}, Target Pseudo Loss: {total_target_loss:.4f}, KL Loss: {total_kl_loss:.4f}, Target Test Accuracy: {test_accuracy:.4f}")


from sklearn.metrics import f1_score

def test(model): 
    model.eval()
    with torch.no_grad():
        tgt_combined, _, _, _ = model(target_features)
        tgt_logits = model.classifier(tgt_combined)
        predictions = tgt_logits.argmax(dim=1)
        accuracy = (predictions == target_labels).float().mean().item()
        
        true_labels = target_labels.cpu().numpy()
        pred_labels = predictions.cpu().numpy()
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')
        
        print(f"Final Target Domain Test Accuracy: {accuracy:.4f}")
        print(f"Final Target Domain Micro-F1: {micro_f1:.4f}")

train(model, optimizer)
test(model)