from load_wm811k import load_wm811k
from collections import Counter

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

# Keep only official Training/Test samples
labeled = df[df["split"].isin(["Training", "Test"])].copy()

def get_shape(x):
    try:
        return (len(x), len(x[0]))
    except Exception:
        return None

labeled["shape"] = labeled["waferMap"].apply(get_shape)

counter = Counter(labeled["shape"])

max_rows = max(s[0] for s in counter if s is not None)
max_cols = max(s[1] for s in counter if s is not None)

print("\n===== LABELED DATASET =====")
print("Total labeled samples:", len(labeled))
print("Training:", (labeled["split"] == "Training").sum())
print("Test:", (labeled["split"] == "Test").sum())

print("\n===== LABELED MAP DIMENSIONS =====")
print("Unique dimensions:", len(counter))
print("Maximum rows:", max_rows)
print("Maximum columns:", max_cols)

print("\n===== LARGEST LABELED MAPS =====")
largest = sorted(
    counter.items(),
    key=lambda x: x[0][0] * x[0][1],
    reverse=True
)

for shape, count in largest[:20]:
    print(f"{shape}: {count}")

print("\n===== MOST COMMON LABELED DIMENSIONS =====")
for shape, count in counter.most_common(20):
    print(f"{shape}: {count}")
