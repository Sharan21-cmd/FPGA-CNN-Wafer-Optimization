import os
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import WaferDataset, CLASSES
from model import WaferCNN


DATA_PATH = "data/raw/LSWMD.pkl"
MODEL_PATH = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# WM-811K LOADER
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


def prepare_dataframe(df):

    df = df.copy()

    df["failure"] = df["failureType"].apply(
        extract_label
    )

    df["split"] = df["trianTestLabel"].apply(
        extract_label
    )

    df = df[
        df["failure"].isin(CLASSES)
        & df["split"].isin(["Training", "Test"])
    ]

    train_df = df[
        df["split"] == "Training"
    ].copy()

    test_df = df[
        df["split"] == "Test"
    ].copy()

    return train_df, test_df


# ============================================================
# FIXED-POINT UTILITIES
# ============================================================

def q_limits(total_bits, signed):

    if signed:

        qmin = -(2 ** (total_bits - 1))
        qmax = (2 ** (total_bits - 1)) - 1

    else:

        qmin = 0
        qmax = (2 ** total_bits) - 1

    return qmin, qmax


def q_scale(integer_bits, fractional_bits, signed):

    return 2.0 ** (-fractional_bits)


def quantize(
    x,
    integer_bits,
    fractional_bits,
    signed=True
):

    total_bits = (
        integer_bits
        + fractional_bits
    )

    scale = q_scale(
        integer_bits,
        fractional_bits,
        signed
    )

    qmin, qmax = q_limits(
        total_bits,
        signed
    )

    q = torch.round(
        x / scale
    )

    q = torch.clamp(
        q,
        qmin,
        qmax
    )

    return q * scale


# ============================================================
# MIXED-Q CNN
# ============================================================

class MixedQCNN(nn.Module):

    def __init__(self, baseline):

        super().__init__()

        # ----------------------------------------------------
        # Copy FP32 model
        # ----------------------------------------------------

        self.conv1 = copy.deepcopy(
            baseline.features[0]
        )

        self.conv2 = copy.deepcopy(
            baseline.features[3]
        )

        self.conv3 = copy.deepcopy(
            baseline.features[6]
        )

        self.fc1 = copy.deepcopy(
            baseline.classifier[1]
        )

        self.fc2 = copy.deepcopy(
            baseline.classifier[4]
        )

        # ----------------------------------------------------
        # Q formats
        # ----------------------------------------------------

        # Input is exactly {0,1,2}.
        self.input_i = 2
        self.input_f = 0

        # Convolution activations.
        self.conv1_i = 4
        self.conv1_f = 4

        self.conv2_i = 4
        self.conv2_f = 4

        self.conv3_i = 5
        self.conv3_f = 3

        # FC1 requires >8 bits for the observed range.
        # Start with 12-bit Q9.3.
        self.fc1_i = 9
        self.fc1_f = 3

        # Output observed range fits Q7.1.
        self.fc2_i = 7
        self.fc2_f = 1

        # ----------------------------------------------------
        # Weight formats
        # ----------------------------------------------------

        # Candidate 8-bit signed weight format.
        self.weight_i = 2
        self.weight_f = 6

        # Quantize weights once.
        with torch.no_grad():

            self.conv1.weight.copy_(
                quantize(
                    self.conv1.weight,
                    self.weight_i,
                    self.weight_f,
                    signed=True
                )
            )

            self.conv2.weight.copy_(
                quantize(
                    self.conv2.weight,
                    self.weight_i,
                    self.weight_f,
                    signed=True
                )
            )

            self.conv3.weight.copy_(
                quantize(
                    self.conv3.weight,
                    self.weight_i,
                    self.weight_f,
                    signed=True
                )
            )

            self.fc1.weight.copy_(
                quantize(
                    self.fc1.weight,
                    self.weight_i,
                    self.weight_f,
                    signed=True
                )
            )

            self.fc2.weight.copy_(
                quantize(
                    self.fc2.weight,
                    self.weight_i,
                    self.weight_f,
                    signed=True
                )
            )

    def forward(self, x):

        # ====================================================
        # INPUT
        # ====================================================

        x = quantize(
            x,
            self.input_i,
            self.input_f,
            signed=False
        )

        # ====================================================
        # CONV 1
        # ====================================================

        x = self.conv1(x)

        x = quantize(
            x,
            self.conv1_i,
            self.conv1_f,
            signed=True
        )

        x = torch.relu(x)

        x = quantize(
            x,
            self.conv1_i,
            self.conv1_f,
            signed=False
        )

        x = torch.max_pool2d(
            x,
            2
        )

        # ====================================================
        # CONV 2
        # ====================================================

        x = self.conv2(x)

        x = quantize(
            x,
            self.conv2_i,
            self.conv2_f,
            signed=True
        )

        x = torch.relu(x)

        x = quantize(
            x,
            self.conv2_i,
            self.conv2_f,
            signed=False
        )

        x = torch.max_pool2d(
            x,
            2
        )

        # ====================================================
        # CONV 3
        # ====================================================

        x = self.conv3(x)

        x = quantize(
            x,
            self.conv3_i,
            self.conv3_f,
            signed=True
        )

        x = torch.relu(x)

        x = quantize(
            x,
            self.conv3_i,
            self.conv3_f,
            signed=False
        )

        x = torch.max_pool2d(
            x,
            2
        )

        # ====================================================
        # FC1
        # ====================================================

        x = torch.flatten(
            x,
            1
        )

        x = self.fc1(x)

        x = quantize(
            x,
            self.fc1_i,
            self.fc1_f,
            signed=True
        )

        x = torch.relu(x)

        x = quantize(
            x,
            self.fc1_i,
            self.fc1_f,
            signed=False
        )

        # ====================================================
        # FC2
        # ====================================================

        x = self.fc2(x)

        x = quantize(
            x,
            self.fc2_i,
            self.fc2_f,
            signed=True
        )

        return x


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    labels_all = []
    predictions_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            labels_all.extend(
                labels.numpy()
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

    labels_all = np.array(
        labels_all
    )

    predictions_all = np.array(
        predictions_all
    )

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0
    )

    return (
        accuracy,
        macro_f1,
        labels_all,
        predictions_all
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("             MIXED-Q FIXED-POINT GOLDEN MODEL")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_wm811k(
        DATA_PATH
    )

    train_df, test_df = prepare_dataframe(
        df
    )

    print()
    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Test samples    :",
        len(test_df)
    )

    train_dataset = WaferDataset(
        train_df,
        image_size=IMAGE_SIZE
    )

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
    # FP32 model
    # --------------------------------------------------------

    baseline = WaferCNN(
        num_classes=9
    ).to(DEVICE)

    baseline.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    baseline.eval()

    fp32_accuracy, fp32_f1, _, _ = evaluate(
        baseline,
        test_loader
    )

    print()
    print("FP32 BASELINE")
    print(
        f"Accuracy : {fp32_accuracy:.6f}"
    )
    print(
        f"Macro F1 : {fp32_f1:.6f}"
    )

    # --------------------------------------------------------
    # Mixed-Q model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MIXED-Q CONFIGURATION")
    print("=" * 70)

    print(
        "Input  : unsigned Q2.0"
    )

    print(
        "Conv1  : signed Q4.4"
    )

    print(
        "Conv2  : signed Q4.4"
    )

    print(
        "Conv3  : signed Q5.3"
    )

    print(
        "FC1    : signed Q9.3 (12-bit)"
    )

    print(
        "FC2    : signed Q7.1"
    )

    print(
        "Weights: signed Q2.6 (8-bit)"
    )

    model = MixedQCNN(
        baseline
    ).to(DEVICE)

    accuracy, macro_f1, labels, predictions = evaluate(
        model,
        test_loader
    )

    print()
    print("=" * 70)
    print("MIXED-Q RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Macro F1 : {macro_f1:.6f}"
    )

    print(
        f"Accuracy drop : "
        f"{fp32_accuracy - accuracy:.6f}"
    )

    print(
        f"Macro F1 drop : "
        f"{fp32_f1 - macro_f1:.6f}"
    )

    print()
    print("PER-CLASS RESULTS")
    print()

    report = classification_report(
        labels,
        predictions,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report_path = (
        "reports/mixed_q_experiment.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "MIXED-Q FIXED-POINT GOLDEN MODEL\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Input: Q2.0 unsigned\n"
        )

        f.write(
            f"Conv1: Q4.4 signed\n"
        )

        f.write(
            f"Conv2: Q4.4 signed\n"
        )

        f.write(
            f"Conv3: Q5.3 signed\n"
        )

        f.write(
            f"FC1: Q9.3 signed (12-bit)\n"
        )

        f.write(
            f"FC2: Q7.1 signed\n"
        )

        f.write(
            f"Weights: Q2.6 signed (8-bit)\n\n"
        )

        f.write(
            f"FP32 Accuracy: {fp32_accuracy:.6f}\n"
        )

        f.write(
            f"FP32 Macro F1: {fp32_f1:.6f}\n\n"
        )

        f.write(
            f"Mixed-Q Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"Mixed-Q Macro F1: {macro_f1:.6f}\n\n"
        )

        f.write(report)

    print(
        "Report saved to:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()
