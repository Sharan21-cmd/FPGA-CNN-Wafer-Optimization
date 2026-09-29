from load_wm811k import load_wm811k
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

PATH = "data/raw/LSWMD.pkl"

print("Loading WM-811K...")
df = load_wm811k(PATH)

def extract_label(x):
    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"

df["failure"] = df["failureType"].apply(extract_label)
df["split"] = df["trianTestLabel"].apply(extract_label)

# Use only official training samples
df = df[df["split"] == "Training"]

classes = [
    "none",
    "Center",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Scratch",
    "Random",
    "Donut",
    "Near-full"
]

sizes = [32, 48, 64]

fig, axes = plt.subplots(
    len(classes),
    len(sizes),
    figsize=(9, 27)
)

for row, cls in enumerate(classes):

    samples = df[df["failure"] == cls]

    if len(samples) == 0:
        continue

    wafer = samples.iloc[0]["waferMap"]

    # Convert wafer map to numpy array
    wafer = np.array(wafer, dtype=np.uint8)

    for col, size in enumerate(sizes):

        # Nearest-neighbor resize
        img = Image.fromarray(wafer)
        resized = img.resize(
            (size, size),
            Image.Resampling.NEAREST
        )

        axes[row, col].imshow(
            resized,
            cmap="gray",
            interpolation="nearest"
        )

        axes[row, col].set_title(
            f"{cls} — {size}×{size}"
        )

        axes[row, col].axis("off")

plt.tight_layout()

output = "reports/input_size_comparison.png"
plt.savefig(output, dpi=200, bbox_inches="tight")

print(f"\nSaved comparison to: {output}")
