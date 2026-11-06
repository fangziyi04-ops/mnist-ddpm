
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Time embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half, device=t.device, dtype=torch.float32
            )
            / (half - 1)
        )

        angles = t.float()[:, None] * frequencies[None, :]

        return torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=1
        )


# 2. Convolution block with time conditioning
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)

        time_signal = self.time_proj(t_emb)
        h = h + time_signal[:, :, None, None]

        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return F.silu(h)


# 3. Small U-Net for MNIST
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_mlp = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)             # 28 x 28
        e2 = self.enc2(self.pool(e1), t_emb) # 14 x 14

        # Bottleneck
        b = self.bottleneck(self.pool(e2), t_emb) # 7 x 7

        # Decoder with skip connections
        d1 = self.up1(b)                     # 14 x 14
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1, t_emb)

        d2 = self.up2(d1)                    # 28 x 28
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2, t_emb)

        return self.output(d2)


# 4. Test model
if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SimpleUNet().to(device)

    x = torch.randn(4, 1, 28, 28, device=device)
    t = torch.randint(0, 1000, (4,), device=device)

    with torch.no_grad():
        output = model(x, t)

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Time shape:", t.shape)
    print("Output shape:", output.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )
    print("Total parameters:", parameters)
    print("Model test successful!")
