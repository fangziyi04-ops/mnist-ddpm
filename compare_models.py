
import os
import torch
import matplotlib.pyplot as plt

from model import SimpleUNet

# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

T = 1000
num_samples = 8

os.makedirs("results", exist_ok=True)

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
# 3. Sampling function
# ==========================================

@torch.no_grad()
def sample(model, initial_noise, step_noises):

    x = initial_noise.clone()

    for t in reversed(range(T)):

        time_tensor = torch.full(
            (num_samples,),
            t,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(x, time_tensor)

        mean = (
            x - betas[t]
            / torch.sqrt(1 - alpha_bars[t])
            * predicted_noise
        ) / torch.sqrt(alphas[t])

        if t > 0:
            x = (
                mean
                + torch.sqrt(posterior_variance[t])
                * step_noises[t - 1]
            )
        else:
            x = mean

    return ((x + 1) / 2).clamp(0, 1)


# ==========================================
# 4. Prepare identical random noise
# ==========================================

generator = torch.Generator(device="cpu")
generator.manual_seed(42)

initial_noise = torch.randn(
    num_samples, 1, 28, 28,
    generator=generator
).to(device)

step_noises = torch.randn(
    T - 1, num_samples, 1, 28, 28,
    generator=generator
).to(device)

# ==========================================
# 5. Compare models
# ==========================================

epochs_to_compare = [10, 20, 30]
all_samples = []

for epoch in epochs_to_compare:

    print(f"Generating with Epoch {epoch} model...")

    model = SimpleUNet().to(device)

    checkpoint_path = (
        f"checkpoints/ddpm_epoch{epoch}.pth"
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    generated = sample(
        model,
        initial_noise,
        step_noises
    )

    all_samples.append(generated.cpu())

    print(f"Epoch {epoch} completed!")

# ==========================================
# 6. Visualization
# ==========================================

fig, axes = plt.subplots(
    3, num_samples,
    figsize=(16, 7)
)

for row, epoch in enumerate(epochs_to_compare):

    for col in range(num_samples):

        axes[row, col].imshow(
            all_samples[row][col, 0],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, col].axis("off")

        if row == 0:
            axes[row, col].set_title(
                f"Sample {col + 1}"
            )

    axes[row, 0].set_ylabel(
        f"Epoch {epoch}",
        fontsize=12
    )

    # Keep the row label visible
    axes[row, 0].yaxis.set_label_coords(-0.15, 0.5)

plt.suptitle(
    "DDPM Generation Comparison: 10 vs 20 vs 30 Epochs",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "results/epoch_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Comparison completed!")
