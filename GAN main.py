import os
import argparse
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

import matplotlib
matplotlib.use("Agg")


class ConvGenerator(nn.Module):
    """
    z -> FC -> (128,7,7) -> ConvT(128->64,k4,s2,p1)+BN+ReLU -> (64,14,14)
       -> ConvT(64->1,k4,s2,p1) + Tanh -> (1,28,28)
    """
    def __init__(self, latent_dim: int = 100, out_ch: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_ch, 4, 2, 1, bias=False),  # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        return self.main(x)


class ConvDiscriminator(nn.Module):
    """Simple conv discriminator producing a single logit."""
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1, bias=False),   # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),     # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = self.main(x)
        h = h.view(x.size(0), -1)
        return self.out(h).squeeze(1)  # raw logit


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_samples(G, device, latent_dim, out_path, num=16, nrow=None):
    G.eval()
    with torch.no_grad():
        z = torch.randn(num, latent_dim, device=device)
        imgs = G(z)
        # rescale from [-1,1] -> [0,1]
        imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1).cpu()
        if nrow is None:
            nrow = int(math.sqrt(num))
        grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vutils.save_image(grid, out_path)


def train_gan(G, D, loader, device, epochs, latent_dim, lr, betas, out_dir,
              sample_every_steps: int = 100):
    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    global_step = 0
    for epoch in range(epochs):
        G.train(); D.train()
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            bsz = real.size(0)

            # ----- Train D -----
            opt_D.zero_grad()
            # real -> 1
            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)
            out_real = D(real)
            d_loss_real = criterion(out_real, real_labels)

            # fake -> 0
            z = torch.randn(bsz, latent_dim, device=device)
            fake = G(z).detach()
            out_fake = D(fake)
            d_loss_fake = criterion(out_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_D.step()

            # ----- Train G -----
            opt_G.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            gen = G(z)
            out = D(gen)
            # want D(gen) -> 1
            g_loss = criterion(out, real_labels)
            g_loss.backward()
            opt_G.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{i}/{len(loader)}] "
                    f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
                )

            # optional: save samples during training
            if sample_every_steps > 0 and (global_step % sample_every_steps == 0):
                sample_path = os.path.join(out_dir, f"samples_e{epoch+1}_s{global_step}.png")
                save_samples(G, device, latent_dim, sample_path, num=16, nrow=4)
            global_step += 1

        # save a sample grid each epoch
        sample_path = os.path.join(out_dir, f"samples_epoch_{epoch+1}.png")
        save_samples(G, device, latent_dim, sample_path, num=16, nrow=4)

    # final checkpoints
    torch.save(G.state_dict(), os.path.join(out_dir, "G_final.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "D_final.pt"))

def scale_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] tensor to [-1,1] for Tanh generator compatibility."""
    return x * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser(description="Train a DCGAN on MNIST (28x28)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=100)  # per your spec
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    # num_workers set to 0 to avoid macOS spawn/pickle issues
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0,1]
        transforms.Lambda(scale_to_minus1_1)   # [-1,1] (no lambda literal)
    ])
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,     # critical for macOS / Windows
        pin_memory=True
    )


    G = ConvGenerator(latent_dim=args.latent_dim, out_ch=1).to(device)
    D = ConvDiscriminator(in_ch=1).to(device)


    train_gan(
        G, D, loader, device,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        out_dir=args.out_dir,
        sample_every_steps=100
    )


    final_grid = os.path.join(args.out_dir, "samples_final.png")
    save_samples(G, device, args.latent_dim, final_grid, num=25, nrow=5)
    print(f"Training complete. Samples saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
