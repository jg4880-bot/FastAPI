
"""
FastAPI image generation API for Module 6:
- GAN (uses ConvGenerator trained in mnist_gan.py)
- Diffusion-like generator (toy implementation)
- Energy-Based Model (toy implementation)
"""

import io
import os
import base64
from typing import List, Literal

import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI
from pydantic import BaseModel
from mnist_gan import ConvGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
GAN_WEIGHTS = "outputs/G_final.pt" 

def load_gan_generator():
    G = ConvGenerator(latent_dim=LATENT_DIM, out_ch=1).to(DEVICE)
    if os.path.exists(GAN_WEIGHTS):
        state_dict = torch.load(GAN_WEIGHTS, map_location=DEVICE)
        G.load_state_dict(state_dict)
        print(f"[GAN] Loaded weights from {GAN_WEIGHTS}")
    else:
        print(f"[GAN] WARNING: {GAN_WEIGHTS} not found, using untrained generator.")
    G.eval()
    return G

GAN_G = load_gan_generator()

class ToyDiffusionGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.net(noise)

DIFFUSION_G = ToyDiffusionGenerator().to(DEVICE)
DIFFUSION_G.eval()

class ToyEnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.Tanh(),   # 28 -> 14
            nn.Conv2d(16, 32, 3, 2, 1), nn.Tanh(),  # 14 -> 7
        )
        self.fc = nn.Linear(32 * 7 * 7, 1)

    def forward(self, x):
        h = self.features(x).view(x.size(0), -1)
        energy = self.fc(h).squeeze(1)
        return energy

EBM = ToyEnergyModel().to(DEVICE)
EBM.eval()

def langevin_sampling(energy_model, n_samples: int, steps: int = 20, step_size: float = 0.1):
    """
    x_{t+1} = x_t - step_size * dE/dx + sqrt(2*step_size)*noise
    """
    x = torch.randn(n_samples, 1, 28, 28, device=DEVICE, requires_grad=True)
    for _ in range(steps):
        energy = energy_model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        x = x - step_size * grad
        x = x + torch.sqrt(torch.tensor(2.0 * step_size, device=DEVICE)) * torch.randn_like(x)
        x = x.detach().requires_grad_(True)
    x = torch.tanh(x)
    return x.detach()

to_pil = transforms.ToPILImage()

def tensor_to_base64(img_tensor: torch.Tensor) -> str:
    """
    img_tensor: (1, 28, 28) or (3, H, W), value in [-1,1] or [0,1]
    return: base64 string
    """
    img = (img_tensor + 1) / 2
    img = torch.clamp(img, 0, 1).cpu()
    pil_img = to_pil(img)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_with_gan(n: int) -> List[str]:
    GAN_G.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        imgs = GAN_G(z)  # (N,1,28,28)
    return [tensor_to_base64(imgs[i]) for i in range(n)]

def generate_with_diffusion(n: int) -> List[str]:
    DIFFUSION_G.eval()
    with torch.no_grad():
        noise = torch.randn(n, 1, 28, 28, device=DEVICE)
        imgs = DIFFUSION_G(noise)  # (N,1,28,28)
    return [tensor_to_base64(imgs[i]) for i in range(n)]

def generate_with_ebm(n: int) -> List[str]:
    EBM.eval()
    imgs = langevin_sampling(EBM, n_samples=n, steps=20, step_size=0.1)
    return [tensor_to_base64(imgs[i]) for i in range(n)]

app = FastAPI(
    title="Module 6 Image Generation API",
    description=(
        "Image generators implemented for Module 6 practical: "
        "GAN, Diffusion-like, and Energy-Based models."
    ),
    version="1.0.0",
)

class GenerateRequest(BaseModel):
    model: Literal["gan", "diffusion", "ebm"] = "gan"
    num_images: int = 4

class GenerateResponse(BaseModel):
    model: str
    num_images: int
    images: List[str] 

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    n = max(1, min(req.num_images, 16)) 
    if req.model == "gan":
        imgs = generate_with_gan(n)
    elif req.model == "diffusion":
        imgs = generate_with_diffusion(n)
    else:
        imgs = generate_with_ebm(n)
    return GenerateResponse(model=req.model, num_images=n, images=imgs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
