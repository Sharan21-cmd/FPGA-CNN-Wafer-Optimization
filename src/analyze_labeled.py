from load_wm811k import load_wm811k
import pandas as pd

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

# Keep only explicitly labeled train/test samples
labeled = df[
    df["split"].isin(["Training", "Test"])
].copy()

print("\n===== LABELED DATASET =====")
print("Total labeled:", len(labeled))

print("\n===== TRAINING BY CLASS =====")
print(
    labeled[labeled["split"] == "Training"]["failure"]
    .value_counts()
    .sort_index()
)

print("\n===== TEST BY CLASS =====")
print(
    labeled[labeled["split"] == "Test"]["failure"]
    .value_counts()
    .sort_index()
)

print("\n===== TRAINING + TEST =====")
print(
    pd.crosstab(
        labeled["failure"],
        labeled["split"]
    )
)

print("\n===== CHECK =====")
print("Training:", (labeled["split"] == "Training").sum())
print("Test:", (labeled["split"] == "Test").sum())
