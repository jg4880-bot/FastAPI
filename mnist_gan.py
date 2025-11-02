import os, math, argparse, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class G(nn.Module):

    def __init__(self, latent_dim=100, out_ch=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128*7*7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 7->14
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, out_ch, 4, 2, 1, bias=False),  # 14->28
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(z.size(0), 128, 7, 7)
        return self.net(x)

class D(nn.Module):

    def __init__(self, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),  # 28->14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)  # 14->7
        )
        self.fc = nn.Linear(128*7*7, 1)
    def forward(self, x):
        h = self.features(x).view(x.size(0), -1)
        return self.fc(h).squeeze(1)  # raw logit

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_samples(Gnet, device, latent_dim, path, n=25, nrow=5):
    Gnet.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        imgs = (Gnet(z) + 1) / 2  # [-1,1] -> [0,1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(imgs.clamp(0,1), path, nrow=nrow, padding=2)

def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*2.0 - 1.0)  # [0,1]->[-1,1] for Tanh
    ])
    ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    Gnet = G(latent_dim=args.latent_dim).to(device)
    Dnet = D().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(Gnet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(Dnet.parameters(), lr=args.lr, betas=(0.5, 0.999))

    steps_per_epoch = math.ceil(len(loader))
    for epoch in range(1, args.epochs+1):
        Gnet.train(); Dnet.train()
        for i, (real, _) in enumerate(loader):
            real = real.to(device); b = real.size(0)
            real_y = torch.ones(b, device=device)
            fake_y = torch.zeros(b, device=device)

          
            optD.zero_grad()
            d_real = criterion(Dnet(real), real_y)
            z = torch.randn(b, args.latent_dim, device=device)
            fake = Gnet(z).detach()
            d_fake = criterion(Dnet(fake), fake_y)
            d_loss = d_real + d_fake
            d_loss.backward(); optD.step()

          
            optG.zero_grad()
            z = torch.randn(b, args.latent_dim, device=device)
            gen = Gnet(z)
            g_loss = criterion(Dnet(gen), real_y)  # fool D
            g_loss.backward(); optG.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] "
                      f"Step [{i}/{steps_per_epoch}] "
                      f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

        
        save_samples(Gnet, device, args.latent_dim,
                     os.path.join(args.out_dir, f"samples_epoch_{epoch}.png"),
                     n=25, nrow=5)

    torch.save(Gnet.state_dict(), os.path.join(args.out_dir, "G_final.pt"))
    torch.save(Dnet.state_dict(), os.path.join(args.out_dir, "D_final.pt"))
    save_samples(Gnet, device, args.latent_dim,
                 os.path.join(args.out_dir, "samples_final.png"),
                 n=25, nrow=5)
    print(f"Done. Outputs in: {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("MNIST DCGAN")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)
