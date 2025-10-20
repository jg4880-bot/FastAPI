import torch

def accuracy(logits, y):
    """Compute batch accuracy."""
    return (logits.argmax(1) == y).float().mean().item()

def train_one_epoch(model, loader, criterion, optim, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = total_acc = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        n = x.size(0)
        total_loss += loss.item() * n
        total_acc  += accuracy(logits, y) * n
        total_n    += n
    return total_loss / total_n, total_acc / total_n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model on validation or test data."""
    model.eval()
    total_loss = total_acc = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        n = x.size(0)
        total_loss += loss.item() * n
        total_acc  += (logits.argmax(1) == y).float().sum().item()
        total_n    += n
    return total_loss / total_n, total_acc / total_n
