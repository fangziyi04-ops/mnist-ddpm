
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ---------------------------------
# 1. Load MNIST dataset
# ---------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

image, label = dataset[0]

print("Image shape:", image.shape)
print("Digit label:", label)

# ---------------------------------
# 2. Define diffusion parameters
# ---------------------------------

T = 1000

betas = torch.linspace(0.0001, 0.02, T)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(alphas, dim=0)

# ---------------------------------
# 3. Forward diffusion function
# ---------------------------------

def forward_diffusion(x0, t, noise=None):

    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar = alpha_bars[t]

    xt = (
        torch.sqrt(alpha_bar) * x0
        + torch.sqrt(1 - alpha_bar) * noise
    )

    return xt

# ---------------------------------
# 4. Visualize diffusion process
# ---------------------------------

timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

fig, axes = plt.subplots(1, len(timesteps), figsize=(16, 3))

noise = torch.randn_like(image)

for i, t in enumerate(timesteps):

    noisy_image = forward_diffusion(image, t, noise)

    axes[i].imshow(
        noisy_image.squeeze().numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    axes[i].set_title(f"t = {t}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("forward_diffusion.png", dpi=200)
plt.show()
