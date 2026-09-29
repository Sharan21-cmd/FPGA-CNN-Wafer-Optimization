from load_wm811k import load_wm811k
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

# Only official labeled data
df = df[df["split"].isin(["Training", "Test"])]

print("\n===== CHECKING WAFER MAP VALUES =====")

value_counts = {}

# Inspect all labeled maps
for wafer in df["waferMap"]:
    arr = np.asarray(wafer)

    for value, count in zip(*np.unique(arr, return_counts=True)):
        value = int(value)
        value_counts[value] = value_counts.get(value, 0) + int(count)

print("\nUnique wafer-map values:")
for value in sorted(value_counts):
    print(f"Value {value}: {value_counts[value]:,} pixels")

print("\nNumber of unique values:", len(value_counts))

print("\n===== SAMPLE MAP =====")
sample = np.asarray(df.iloc[0]["waferMap"])

print("Shape:", sample.shape)
print("Dtype:", sample.dtype)
print("Unique values:", np.unique(sample))
print(sample)
