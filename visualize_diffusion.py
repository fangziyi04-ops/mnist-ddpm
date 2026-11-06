
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from model import SimpleUNet

# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

T = 1000
torch.manual_seed(42)

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

print("Model loaded successfully!")

# ==========================================
# 4. Reverse diffusion and save frames
# ==========================================

frames = []
saved_steps = []

@torch.no_grad()
def generate():

    x = torch.randn(1, 1, 28, 28, device=device)

    # Capture every 20 steps
    capture_steps = set(range(0, T, 20))
    capture_steps.add(T - 1)

    for t in reversed(range(T)):

        time_tensor = torch.full(
            (1,), t,
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
            noise = torch.randn_like(x)
            x = (
                mean
                + torch.sqrt(posterior_variance[t])
                * noise
            )
        else:
            x = mean

        if t in capture_steps:
            image = (
                (x[0, 0] + 1) / 2
            ).clamp(0, 1)

            image = (
                image.cpu().numpy() * 255
            ).astype("uint8")

            frame = Image.fromarray(image).resize(
                (280, 280),
                resample=Image.Resampling.NEAREST
            )

            frames.append(frame)
            saved_steps.append(t)

            if t % 100 == 0:
                print(f"Saved timestep: {t}")

    return x


# ==========================================
# 5. Generate animation
# ==========================================

generate()

frames[0].save(
    "results/reverse_diffusion.gif",
    save_all=True,
    append_images=frames[1:],
    duration=120,
    loop=0
)

print("GIF saved successfully!")

# ==========================================
# 6. Save selected stages
# ==========================================

target_steps = [999, 800, 600, 400, 200, 0]

fig, axes = plt.subplots(
    1, len(target_steps),
    figsize=(14, 3)
)

for i, target in enumerate(target_steps):

    index = min(
        range(len(saved_steps)),
        key=lambda j: abs(saved_steps[j] - target)
    )

    axes[i].imshow(frames[index], cmap="gray")
    axes[i].set_title(f"t = {saved_steps[index]}")
    axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    "results/reverse_diffusion_stages.png",
    dpi=200
)

plt.show()

print("Visualization completed!")
