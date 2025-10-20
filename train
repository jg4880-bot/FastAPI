import torch
import torch.nn as nn
from torch.optim import Adam
from cnn.config import EPOCHS, LR, WEIGHTS_PATH
from cnn.data import cifar10_loaders
from cnn.model import SimpleCNN
from cnn.engine import train_one_epoch, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = cifar10_loaders()

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=LR)

    best_val, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {ep:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test {test_loss:.4f}/{test_acc:.4f}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, WEIGHTS_PATH)
    print(f"Saved -> {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()

