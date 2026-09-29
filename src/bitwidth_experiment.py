import os
import time
import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import WaferDataset, CLASSES
from model import WaferCNN


DATA_PATH = "data/raw/LSWMD.pkl"
BASELINE_MODEL = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LEGACY PICKLE LOADER
# ============================================================

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

    print("Loading WM-811K...")

    with open(path, "rb") as f:
        return CustomUnpickler(
            f,
            encoding="latin1"
        ).load()


def extract_label(x):

    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["failure"] = df["failureType"].apply(
        extract_label
    )

    df["split"] = df["trianTestLabel"].apply(
        extract_label
    )

    valid_classes = set(CLASSES)

    df = df[
        df["failure"].isin(valid_classes)
        & df["split"].isin(["Training", "Test"])
    ].copy()

    train_df = df[
        df["split"] == "Training"
    ].copy()

    test_df = df[
        df["split"] == "Test"
    ].copy()

    return train_df, test_df


# ============================================================
# BIT REPRESENTATION
# ============================================================

def apply_bitwidth(image, bits):

    """
    Simulate unsigned integer representation
    using the requested number of bits.

    Since the original wafer values are only:
        0, 1, 2

    2 bits are sufficient to represent all values
    without information loss.

    This function does NOT claim to physically make
    PyTorch use a 2-bit CPU datatype.
    """

    levels = (1 << bits) - 1

    image = torch.clamp(
        image,
        0,
        levels
    )

    return image


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, bits):

    model.eval()

    all_labels = []
    all_predictions = []

    start = time.perf_counter()

    with torch.no_grad():

        for images, labels in loader:

            images = apply_bitwidth(
                images,
                bits
            )

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    labels_np = np.array(all_labels)
    predictions_np = np.array(
        all_predictions
    )

    accuracy = accuracy_score(
        labels_np,
        predictions_np
    )

    macro_f1 = f1_score(
        labels_np,
        predictions_np,
        average="macro",
        zero_division=0
    )

    latency = (
        elapsed
        / len(labels_np)
    ) * 1000

    return accuracy, macro_f1, latency


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("       INPUT BIT-WIDTH EXPERIMENT")
    print("========================================")

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_wm811k(DATA_PATH)

    train_df, test_df = prepare_dataframe(df)

    print("\nTraining samples:", len(train_df))
    print("Test samples:", len(test_df))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = WaferDataset(
        test_df,
        image_size=IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load baseline model
    # --------------------------------------------------------

    model = WaferCNN(
        num_classes=len(CLASSES)
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            BASELINE_MODEL,
            map_location=DEVICE
        )
    )

    print("\nBaseline model loaded:")
    print(BASELINE_MODEL)

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    bit_widths = [8, 4, 2]

    results = []

    print("\n===== RUNNING EXPERIMENTS =====")

    for bits in bit_widths:

        print(
            f"\nTesting {bits}-bit representation..."
        )

        accuracy, macro_f1, latency = evaluate(
            model,
            test_loader,
            bits
        )

        print(
            f"Bits      : {bits}"
        )

        print(
            f"Accuracy  : {accuracy:.6f}"
        )

        print(
            f"Macro F1  : {macro_f1:.6f}"
        )

        print(
            f"Latency   : {latency:.6f} ms/image"
        )

        results.append(
            (
                bits,
                accuracy,
                macro_f1,
                latency
            )
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report_path = (
        "reports/bitwidth_experiment.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "INPUT BIT-WIDTH EXPERIMENT\n"
        )

        f.write(
            "==========================\n\n"
        )

        f.write(
            "Dataset: WM-811K\n"
        )

        f.write(
            "Input size: 48x48\n"
        )

        f.write(
            f"Test samples: {len(test_df)}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "This software experiment evaluates "
            "the effect of representing the existing "
            "wafer values with different bit widths. "
            "It does not claim that PyTorch CPU "
            "execution physically uses 2-bit or 4-bit "
            "arithmetic.\n\n"
        )

        f.write(
            "RESULTS\n"
        )

        f.write(
            "-------\n"
        )

        f.write(
            "Bits | Accuracy | Macro F1 | Latency ms/image\n"
        )

        for bits, accuracy, macro_f1, latency in results:

            f.write(
                f"{bits:4d} | "
                f"{accuracy:.6f} | "
                f"{macro_f1:.6f} | "
                f"{latency:.6f}\n"
            )

    print("\n========================================")
    print("Experiment complete.")
    print("Report saved to:")
    print(report_path)
    print("========================================")


if __name__ == "__main__":
    main()
