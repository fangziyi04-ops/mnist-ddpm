
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SimpleUNet

# Configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

T = 1000
batch_size = 64
target_epochs = 30
learning_rate = 0.0002

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Dataset
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

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# Diffusion schedule
betas = torch.linspace(
    0.0001, 0.02, T, device=device
)

alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# Model
model = SimpleUNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# Load checkpoint
checkpoint = torch.load(
    "checkpoints/ddpm_latest.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"]
loss_history = checkpoint["loss_history"]

print(f"Resuming from epoch {start_epoch}")
print("Device:", device)

# Continue training
for epoch in range(start_epoch, target_epochs):

    model.train()
    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{target_epochs}"
    )

    for images, _ in progress:

        images = images.to(
            device, non_blocking=True
        )

        t = torch.randint(
            0, T,
            (images.size(0),),
            device=device
        )

        noise = torch.randn_like(images)

        alpha_bar = alpha_bars[t].view(
            -1, 1, 1, 1
        )

        noisy_images = (
            torch.sqrt(alpha_bar) * images
            + torch.sqrt(1 - alpha_bar) * noise
        )

        predicted_noise = model(
            noisy_images, t
        )

        loss = F.mse_loss(
            predicted_noise, noise
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    print(
        f"Epoch {epoch + 1}/{target_epochs} "
        f"- Average Loss: {avg_loss:.6f}"
    )

    # Save checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history
        },
        "checkpoints/ddpm_latest.pth"
    )

    # Save model at 20 and 30 epochs
    if (epoch + 1) in [20, 30]:
        torch.save(
            model.state_dict(),
            f"checkpoints/ddpm_epoch{epoch + 1}.pth"
        )

# Save final model
torch.save(
    model.state_dict(),
    "checkpoints/ddpm_final.pth"
)

# Plot full loss history
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("DDPM Training Loss (30 Epochs)")
plt.grid(True)
plt.savefig("results/training_loss_30epochs.png", dpi=200)
plt.show()

print("Training completed!")
