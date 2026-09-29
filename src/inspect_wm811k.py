from load_wm811k import load_wm811k

PATH = "data/raw/LSWMD.pkl"

print("Loading WM-811K...")
df = load_wm811k(PATH)

print("\n===== DATASET =====")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


def extract_label(x):
    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


failure_labels = df["failureType"].apply(extract_label)
split_labels = df["trianTestLabel"].apply(extract_label)


print("\n===== FAILURE TYPE DISTRIBUTION =====")
print(failure_labels.value_counts())


print("\n===== TRAIN / TEST DISTRIBUTION =====")
print(split_labels.value_counts())


print("\n===== DIE SIZE =====")
print(df["dieSize"].describe())


def get_shape(x):
    try:
        return (len(x), len(x[0]))
    except Exception:
        return None


map_shapes = df["waferMap"].apply(get_shape)

print("\n===== COMMON WAFER MAP SIZES =====")
print(map_shapes.value_counts().head(20))


print("\n===== DEFECT CLASSES (EXCLUDING NONE) =====")
print(failure_labels[failure_labels != "none"].value_counts())


print("\n===== EXAMPLE LABELS =====")
print("First 10 failure types:")
print(failure_labels.head(10).tolist())

print("\nFirst 10 train/test labels:")
print(split_labels.head(10).tolist())
