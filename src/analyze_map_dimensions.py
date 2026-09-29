from load_wm811k import load_wm811k
from collections import Counter

PATH = "data/raw/LSWMD.pkl"

print("Loading WM-811K...")
df = load_wm811k(PATH)

def get_shape(x):
    try:
        return (len(x), len(x[0]))
    except Exception:
        return None

shapes = df["waferMap"].apply(get_shape)

counter = Counter(shapes)

print("\n===== MAP DIMENSION STATISTICS =====")
print("Number of unique dimensions:", len(counter))

max_rows = max(s[0] for s in counter if s is not None)
max_cols = max(s[1] for s in counter if s is not None)

print("Maximum rows:", max_rows)
print("Maximum columns:", max_cols)

print("\n===== LARGEST MAP DIMENSIONS =====")
largest = sorted(
    counter.items(),
    key=lambda x: x[0][0] * x[0][1],
    reverse=True
)

for shape, count in largest[:20]:
    print(f"{shape}: {count}")

print("\n===== MOST COMMON DIMENSIONS =====")
for shape, count in counter.most_common(20):
    print(f"{shape}: {count}")
