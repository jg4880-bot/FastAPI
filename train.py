import torch
import torch.nn as nn
from torch.optim import Adam

from config import EPOCHS, LR, WEIGHTS_PATH
from data import cifar10_loaders
from model import SimpleCNN
from engine import train_one_epoch, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = cifar10_loaders()

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optim, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test {test_loss:.4f}/{test_acc:.4f}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, WEIGHTS_PATH)
    print(f"Saved -> {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()

