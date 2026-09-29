import pickle
import pandas as pd

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace(
                "pandas.indexes",
                "pandas.core.indexes",
                1
            )
        return super().find_class(module, name)

def load_wm811k(path):
    with open(path, "rb") as f:
        return CustomUnpickler(f, encoding="latin1").load()

if __name__ == "__main__":
    path = "data/raw/LSWMD.pkl"

    print("Loading WM-811K...")
    print("This may take a while because the file is ~2 GB.")

    df = load_wm811k(path)

    print("\n===== WM-811K LOADED =====")
    print("Type:", type(df))
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())
