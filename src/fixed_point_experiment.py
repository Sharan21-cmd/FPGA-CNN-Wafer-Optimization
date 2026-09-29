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
CALIBRATION_BATCHES = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA LOADING
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
    ].copy()

    train_df = df[
        df["split"] == "Training"
    ].copy()

    test_df = df[
        df["split"] == "Test"
    ].copy()

    return train_df, test_df


# ============================================================
# QUANTIZATION
# ============================================================

def get_qrange(bits, signed=True):

    if signed:

        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

    else:

        qmin = 0
        qmax = (2 ** bits) - 1

    return qmin, qmax


def get_scale(max_abs, bits, signed=True):

    _, qmax = get_qrange(
        bits,
        signed
    )

    if max_abs <= 0:
        return 1.0

    return max_abs / qmax


def fake_quant(
    tensor,
    scale,
    bits,
    signed=True
):

    qmin, qmax = get_qrange(
        bits,
        signed
    )

    if scale <= 0:
        return tensor

    integer = torch.round(
        tensor / scale
    )

    integer = torch.clamp(
        integer,
        qmin,
        qmax
    )

    return integer * scale


# ============================================================
# CALIBRATION
# ============================================================

class RangeCollector:

    def __init__(self):

        self.maximums = {}

    def hook(self, name):

        def capture(module, inputs, output):

            value = output.detach()

            maximum = value.abs().max().item()

            if name not in self.maximums:

                self.maximums[name] = maximum

            else:

                self.maximums[name] = max(
                    self.maximums[name],
                    maximum
                )

        return capture


def calibrate(model, loader):

    collector = RangeCollector()

    handles = []

    for name, module in model.named_modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ReLU,
                nn.MaxPool2d,
                nn.Linear
            )
        ):

            handles.append(
                module.register_forward_hook(
                    collector.hook(name)
                )
            )

    model.eval()

    with torch.no_grad():

        for index, (images, labels) in enumerate(loader):

            if index >= CALIBRATION_BATCHES:
                break

            images = images.to(DEVICE)

            model(images)

    for handle in handles:
        handle.remove()

    return collector.maximums


# ============================================================
# QUANTIZED MODEL
# ============================================================

class QuantizedCNN(nn.Module):

    def __init__(
        self,
        baseline,
        bits,
        ranges
    ):

        super().__init__()

        self.bits = bits

        # Copy original parameters.
        self.conv1 = copy.deepcopy(
            baseline.features[0]
        )

        self.relu1 = baseline.features[1]

        self.pool1 = baseline.features[2]

        self.conv2 = copy.deepcopy(
            baseline.features[3]
        )

        self.relu2 = baseline.features[4]

        self.pool2 = baseline.features[5]

        self.conv3 = copy.deepcopy(
            baseline.features[6]
        )

        self.relu3 = baseline.features[7]

        self.pool3 = baseline.features[8]

        self.fc1 = copy.deepcopy(
            baseline.classifier[1]
        )

        self.relu4 = baseline.classifier[2]

        self.dropout = baseline.classifier[3]

        self.fc2 = copy.deepcopy(
            baseline.classifier[4]
        )

        # ----------------------------------------------------
        # Weight scales
        # ----------------------------------------------------

        self.w1_scale = get_scale(
            self.conv1.weight.detach().abs().max().item(),
            bits,
            signed=True
        )

        self.w2_scale = get_scale(
            self.conv2.weight.detach().abs().max().item(),
            bits,
            signed=True
        )

        self.w3_scale = get_scale(
            self.conv3.weight.detach().abs().max().item(),
            bits,
            signed=True
        )

        self.fc1_scale = get_scale(
            self.fc1.weight.detach().abs().max().item(),
            bits,
            signed=True
        )

        self.fc2_scale = get_scale(
            self.fc2.weight.detach().abs().max().item(),
            bits,
            signed=True
        )

        # ----------------------------------------------------
        # Quantize weights
        # ----------------------------------------------------

        with torch.no_grad():

            self.conv1.weight.copy_(
                fake_quant(
                    self.conv1.weight,
                    self.w1_scale,
                    bits,
                    signed=True
                )
            )

            self.conv2.weight.copy_(
                fake_quant(
                    self.conv2.weight,
                    self.w2_scale,
                    bits,
                    signed=True
                )
            )

            self.conv3.weight.copy_(
                fake_quant(
                    self.conv3.weight,
                    self.w3_scale,
                    bits,
                    signed=True
                )
            )

            self.fc1.weight.copy_(
                fake_quant(
                    self.fc1.weight,
                    self.fc1_scale,
                    bits,
                    signed=True
                )
            )

            self.fc2.weight.copy_(
                fake_quant(
                    self.fc2.weight,
                    self.fc2_scale,
                    bits,
                    signed=True
                )
            )

        # ----------------------------------------------------
        # Activation scales
        # ----------------------------------------------------

        self.s_conv1 = get_scale(
            ranges["features.0"],
            bits,
            signed=True
        )

        self.s_relu1 = get_scale(
            ranges["features.1"],
            bits,
            signed=False
        )

        self.s_pool1 = get_scale(
            ranges["features.2"],
            bits,
            signed=False
        )

        self.s_conv2 = get_scale(
            ranges["features.3"],
            bits,
            signed=True
        )

        self.s_relu2 = get_scale(
            ranges["features.4"],
            bits,
            signed=False
        )

        self.s_pool2 = get_scale(
            ranges["features.5"],
            bits,
            signed=False
        )

        self.s_conv3 = get_scale(
            ranges["features.6"],
            bits,
            signed=True
        )

        self.s_relu3 = get_scale(
            ranges["features.7"],
            bits,
            signed=False
        )

        self.s_pool3 = get_scale(
            ranges["features.8"],
            bits,
            signed=False
        )

        self.s_fc1 = get_scale(
            ranges["classifier.1"],
            bits,
            signed=True
        )

        self.s_relu4 = get_scale(
            ranges["classifier.2"],
            bits,
            signed=False
        )

        self.s_fc2 = get_scale(
            ranges["classifier.4"],
            bits,
            signed=True
        )

    def forward(self, x):

        # Input: original wafer values 0,1,2.
        # Exact 2-bit unsigned representation.
        x = fake_quant(
            x,
            1.0,
            2,
            signed=False
        )

        # ----------------------------------------------------
        # Conv 1
        # ----------------------------------------------------

        x = self.conv1(x)

        x = fake_quant(
            x,
            self.s_conv1,
            self.bits,
            signed=True
        )

        x = self.relu1(x)

        x = fake_quant(
            x,
            self.s_relu1,
            self.bits,
            signed=False
        )

        x = self.pool1(x)

        x = fake_quant(
            x,
            self.s_pool1,
            self.bits,
            signed=False
        )

        # ----------------------------------------------------
        # Conv 2
        # ----------------------------------------------------

        x = self.conv2(x)

        x = fake_quant(
            x,
            self.s_conv2,
            self.bits,
            signed=True
        )

        x = self.relu2(x)

        x = fake_quant(
            x,
            self.s_relu2,
            self.bits,
            signed=False
        )

        x = self.pool2(x)

        x = fake_quant(
            x,
            self.s_pool2,
            self.bits,
            signed=False
        )

        # ----------------------------------------------------
        # Conv 3
        # ----------------------------------------------------

        x = self.conv3(x)

        x = fake_quant(
            x,
            self.s_conv3,
            self.bits,
            signed=True
        )

        x = self.relu3(x)

        x = fake_quant(
            x,
            self.s_relu3,
            self.bits,
            signed=False
        )

        x = self.pool3(x)

        x = fake_quant(
            x,
            self.s_pool3,
            self.bits,
            signed=False
        )

        # ----------------------------------------------------
        # Fully connected
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            1
        )

        x = self.fc1(x)

        x = fake_quant(
            x,
            self.s_fc1,
            self.bits,
            signed=True
        )

        x = self.relu4(x)

        x = fake_quant(
            x,
            self.s_relu4,
            self.bits,
            signed=False
        )

        x = self.dropout(x)

        x = self.fc2(x)

        x = fake_quant(
            x,
            self.s_fc2,
            self.bits,
            signed=True
        )

        return x


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

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

    labels = np.array(
        all_labels
    )

    predictions = np.array(
        all_predictions
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return (
        accuracy,
        macro_f1,
        labels,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       CORRECTED FIXED-POINT EXPERIMENT")
    print("=" * 70)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load dataset
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

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_dataset = WaferDataset(
        train_df,
        image_size=IMAGE_SIZE
    )

    test_dataset = WaferDataset(
        test_df,
        image_size=IMAGE_SIZE
    )

    calibration_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = WaferCNN(
        num_classes=len(CLASSES)
    ).to(DEVICE)

    baseline.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    baseline.eval()

    print()
    print(
        "Loaded baseline:",
        MODEL_PATH
    )

    baseline_accuracy, baseline_f1, _, _ = evaluate(
        baseline,
        test_loader
    )

    print()
    print("FP32 BASELINE")
    print(
        f"Accuracy : {baseline_accuracy:.6f}"
    )
    print(
        f"Macro F1 : {baseline_f1:.6f}"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CALIBRATING ACTIVATION RANGES")
    print("=" * 70)

    ranges = calibrate(
        baseline,
        calibration_loader
    )

    for name, value in ranges.items():

        print(
            f"{name:25s}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Precision experiments
    # --------------------------------------------------------

    bit_widths = [
        8,
        7,
        6,
        5,
        4
    ]

    results = []

    for bits in bit_widths:

        print()
        print("=" * 70)
        print(
            f"TESTING {bits}-BIT MODEL"
        )
        print("=" * 70)

        model = QuantizedCNN(
            baseline,
            bits,
            ranges
        ).to(DEVICE)

        accuracy, macro_f1, labels, predictions = evaluate(
            model,
            test_loader
        )

        print()
        print(
            f"{bits}-BIT RESULTS"
        )

        print(
            f"Accuracy : {accuracy:.6f}"
        )

        print(
            f"Macro F1 : {macro_f1:.6f}"
        )

        print(
            f"Accuracy drop : "
            f"{baseline_accuracy - accuracy:.6f}"
        )

        print(
            f"Macro F1 drop : "
            f"{baseline_f1 - macro_f1:.6f}"
        )

        results.append(
            (
                bits,
                accuracy,
                macro_f1
            )
        )

        # ----------------------------------------------------
        # Per-class report
        # ----------------------------------------------------

        print()
        print(
            f"PER-CLASS RESULTS ({bits}-BIT)"
        )

        print(
            classification_report(
                labels,
                predictions,
                target_names=CLASSES,
                digits=4,
                zero_division=0
            )
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report_path = (
        "reports/fixed_point_experiment_corrected.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "CORRECTED FIXED-POINT EXPERIMENT\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Input size: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Training samples: {len(train_df)}\n"
        )

        f.write(
            f"Test samples: {len(test_df)}\n\n"
        )

        f.write(
            "FP32 BASELINE\n"
        )

        f.write(
            f"Accuracy: {baseline_accuracy:.6f}\n"
        )

        f.write(
            f"Macro F1: {baseline_f1:.6f}\n\n"
        )

        f.write(
            "RESULTS\n"
        )

        f.write(
            "Bits | Accuracy | Macro F1 | "
            "Accuracy Drop | F1 Drop\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        for bits, accuracy, macro_f1 in results:

            f.write(
                f"{bits:4d} | "
                f"{accuracy:.6f} | "
                f"{macro_f1:.6f} | "
                f"{baseline_accuracy - accuracy:.6f} | "
                f"{baseline_f1 - macro_f1:.6f}\n"
            )

    print()
    print("=" * 70)
    print("CORRECTED EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print(
        "Report saved to:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()
