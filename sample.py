
import os
import torch
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid
from model import SimpleUNet

# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

T = 1000
num_samples = 16

os.makedirs("results", exist_ok=True)

torch.manual_seed(42)

# ==========================================
# 2. Diffusion schedule
# ==========================================

betas = torch.linspace(
    0.0001, 0.02, T, device=device
)

alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

alpha_bars_prev = torch.cat([
    torch.ones(1, device=device),
    alpha_bars[:-1]
])

posterior_variance = (
    betas * (1 - alpha_bars_prev)
    / (1 - alpha_bars)
)

# ==========================================
# 3. Load trained model
# ==========================================

model = SimpleUNet().to(device)

state_dict = torch.load(
    "checkpoints/ddpm_final.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Trained model loaded!")

# ==========================================
# 4. Reverse diffusion
# ==========================================

@torch.no_grad()
def sample():

    # Start from pure Gaussian noise
    x = torch.randn(
        num_samples, 1, 28, 28,
        device=device
    )

    for t in reversed(range(T)):

        time_tensor = torch.full(
            (num_samples,),
            t,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(x, time_tensor)

        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        beta = betas[t]

        # Predicted reverse mean
        mean = (
            x - beta
            / torch.sqrt(1 - alpha_bar)
            * predicted_noise
        ) / torch.sqrt(alpha)

        if t > 0:
            noise = torch.randn_like(x)
            x = (
                mean
                + torch.sqrt(posterior_variance[t])
                * noise
            )
        else:
            x = mean

        if t % 100 == 0:
            print(f"Sampling timestep: {t}")

    return x


# ==========================================
# 5. Generate images
# ==========================================

samples = sample()

# Convert [-1, 1] to [0, 1]
samples = ((samples + 1) / 2).clamp(0, 1)

save_image(
    samples,
    "results/generated_digits.png",
    nrow=4
)

grid = make_grid(samples, nrow=4)

plt.figure(figsize=(7, 7))
plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("MNIST DDPM Generated Samples")
plt.tight_layout()
plt.savefig("results/generated_digits_preview.png", dpi=200)
plt.show()

print("Sampling completed!")
