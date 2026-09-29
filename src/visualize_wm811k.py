from load_wm811k import load_wm811k
import matplotlib.pyplot as plt

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

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, cls in zip(axes.flat, classes):
    samples = df[
        (df["failure"] == cls) &
        (df["split"] == "Training")
    ]

    if len(samples) == 0:
        ax.set_title(f"{cls} - no sample")
        ax.axis("off")
        continue

    wafer = samples.iloc[0]["waferMap"]

    ax.imshow(wafer, cmap="gray")
    ax.set_title(cls)
    ax.axis("off")

plt.tight_layout()

output = "reports/wafer_classes.png"
plt.savefig(output, dpi=200)

print(f"\nSaved visualization to: {output}")
