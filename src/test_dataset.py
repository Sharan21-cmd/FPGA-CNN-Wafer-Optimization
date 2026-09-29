from load_wm811k import load_wm811k
from dataset import WaferDataset

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

train_df = df[df["split"] == "Training"].copy()
test_df = df[df["split"] == "Test"].copy()

print("\nTraining samples:", len(train_df))
print("Test samples:", len(test_df))

train_dataset = WaferDataset(train_df, image_size=48)

image, label = train_dataset[0]

print("\n===== DATASET CHECK =====")
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Label:", label.item())
print("Image min:", image.min().item())
print("Image max:", image.max().item())
print("Unique values:", image.unique().tolist())
