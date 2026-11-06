
import os
import csv
import torch
import matplotlib.pyplot as plt

from model import SimpleUNet
from classifier import MNISTClassifier


# ==========================================
# 1. Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

T = 1000
num_samples = 500
batch_size = 50
epochs_to_compare = [10, 20, 30]

os.makedirs("results", exist_ok=True)

print("Evaluation device:", device)


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
# 3. Load independent classifier
# ==========================================

classifier = MNISTClassifier().to(device)

classifier.load_state_dict(
    torch.load(
        "checkpoints/mnist_classifier.pth",
        map_location=device,
        weights_only=True
    )
)

classifier.eval()

print("Classifier loaded!")


# ==========================================
# 4. DDPM sampling
# ==========================================

@torch.inference_mode()
def generate_samples(model, n, generator):

    x = torch.randn(
        n, 1, 28, 28,
        device=device,
        generator=generator
    )

    for t in reversed(range(T)):

        time_tensor = torch.full(
            (n,), t,
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
            noise = torch.randn(
                x.shape,
                device=device,
                generator=generator
            )

            x = (
                mean
                + torch.sqrt(posterior_variance[t])
                * noise
            )
        else:
            x = mean

    # Classifier expects images normalized to [-1, 1]
    return x.clamp(-1, 1)


# ==========================================
# 5. Evaluate checkpoints
# ==========================================

results = []

for epoch in epochs_to_compare:

    print(f"\nEvaluating Epoch {epoch}...")

    model = SimpleUNet().to(device)

    path = f"checkpoints/ddpm_epoch{epoch}.pth"

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Reset seed for fair comparisons
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    all_confidences = []
    all_predictions = []

    for start in range(0, num_samples, batch_size):

        current_batch = min(
            batch_size, num_samples - start
        )

        generated = generate_samples(
            model, current_batch, generator
        )

        with torch.inference_mode():

            logits = classifier(generated)

            probabilities = torch.softmax(
                logits, dim=1
            )

            confidence, predictions = (
                probabilities.max(dim=1)
            )

        all_confidences.append(
            confidence.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        print(
            f"Generated {start + current_batch}"
            f"/{num_samples}"
        )

    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)

    mean_confidence = confidences.mean().item()

    high_confidence_rate = (
        (confidences >= 0.9)
        .float().mean().item()
    )

    counts = torch.bincount(
        predictions, minlength=10
    )

    class_coverage = (
        (counts > 0).sum().item()
    )

    results.append({
        "epoch": epoch,
        "mean_confidence": mean_confidence,
        "high_confidence_rate": high_confidence_rate,
        "class_coverage": class_coverage,
        "counts": counts.tolist()
    })

    print(f"Mean confidence: {mean_confidence:.4f}")
    print(f"High-confidence rate: {high_confidence_rate:.2%}")
    print(f"Class coverage: {class_coverage}/10")


# ==========================================
# 6. Save CSV
# ==========================================

with open(
    "results/evaluation_metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Mean Confidence",
        "High Confidence Rate",
        "Class Coverage"
    ])

    for result in results:

        writer.writerow([
            result["epoch"],
            result["mean_confidence"],
            result["high_confidence_rate"],
            result["class_coverage"]
        ])


# ==========================================
# 7. Plot metrics
# ==========================================

epochs = [r["epoch"] for r in results]

confidences = [
    r["mean_confidence"] for r in results
]

rates = [
    r["high_confidence_rate"] for r in results
]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    confidences,
    marker="o",
    label="Mean Confidence"
)

plt.plot(
    epochs,
    rates,
    marker="s",
    label="High-confidence Rate"
)

plt.xlabel("Training Epoch")
plt.ylabel("Score")
plt.title("DDPM Quantitative Evaluation")
plt.xticks(epochs)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.savefig(
    "results/evaluation_metrics.png",
    dpi=200
)

plt.close()


# ==========================================
# 8. Plot class distribution
# ==========================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(9, 10)
)

for i, result in enumerate(results):

    axes[i].bar(
        range(10),
        result["counts"]
    )

    axes[i].set_title(
        f"Epoch {result['epoch']}"
    )

    axes[i].set_xlabel("Predicted Digit")
    axes[i].set_ylabel("Number of Samples")
    axes[i].set_xticks(range(10))
    axes[i].set_ylim(0, num_samples)

plt.tight_layout()

plt.savefig(
    "results/class_distribution.png",
    dpi=200
)

plt.close()

print("\nEvaluation completed!")
