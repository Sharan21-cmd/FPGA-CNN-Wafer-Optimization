import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WaferDataset
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
# LEGACY WM-811K LOADER
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

    classes = [
        "none",
        "Center",
        "Edge-Loc",
        "Edge-Ring",
        "Loc",
        "Scratch",
        "Random",
        "Donut",
        "Near-full"
    ]

    df = df[
        df["failure"].isin(classes)
        & df["split"].isin(["Training", "Test"])
    ]

    return df[
        df["split"] == "Training"
    ].copy()


# ============================================================
# ACTIVATION RANGE COLLECTION
# ============================================================

class ActivationCollector:

    def __init__(self):

        self.maximum = {}
        self.minimum = {}

    def hook(self, name):

        def capture(module, inputs, output):

            x = output.detach()

            current_min = x.min().item()
            current_max = x.max().item()

            if name not in self.maximum:

                self.minimum[name] = current_min
                self.maximum[name] = current_max

            else:

                self.minimum[name] = min(
                    self.minimum[name],
                    current_min
                )

                self.maximum[name] = max(
                    self.maximum[name],
                    current_max
                )

        return capture


def collect_activation_ranges(
    model,
    loader
):

    collector = ActivationCollector()

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

        for batch_index, (images, _) in enumerate(loader):

            if batch_index >= CALIBRATION_BATCHES:
                break

            images = images.to(DEVICE)

            model(images)

    for handle in handles:

        handle.remove()

    return collector.minimum, collector.maximum


# ============================================================
# INTEGER BIT REQUIREMENT
# ============================================================

def unsigned_integer_bits(max_value):

    if max_value <= 0:

        return 1

    return int(
        np.ceil(
            np.log2(max_value + 1)
        )
    )


def signed_integer_bits(min_value, max_value):

    maximum = max(
        abs(min_value),
        abs(max_value)
    )

    if maximum <= 0:

        return 1

    # One sign bit + magnitude bits.
    return 1 + int(
        np.ceil(
            np.log2(maximum + 1)
        )
    )


def analyze_format(
    minimum,
    maximum,
    total_bits,
    signed
):

    if signed:

        integer_bits = signed_integer_bits(
            minimum,
            maximum
        )

        # Leave at least one fractional bit
        # whenever possible.
        fractional_bits = max(
            0,
            total_bits - integer_bits
        )

        representable_max = (
            (2 ** (total_bits - 1) - 1)
            / (2 ** fractional_bits)
        )

        representable_min = (
            -(2 ** (total_bits - 1))
            / (2 ** fractional_bits)
        )

    else:

        integer_bits = unsigned_integer_bits(
            maximum
        )

        fractional_bits = max(
            0,
            total_bits - integer_bits
        )

        representable_min = 0

        representable_max = (
            (2 ** total_bits - 1)
            / (2 ** fractional_bits)
        )

    fits = (
        representable_min <= minimum
        and
        representable_max >= maximum
    )

    return (
        integer_bits,
        fractional_bits,
        representable_min,
        representable_max,
        fits
    )


# ============================================================
# WEIGHT ANALYSIS
# ============================================================

def get_weight_range(
    tensor
):

    x = tensor.detach().cpu().numpy()

    return (
        float(x.min()),
        float(x.max())
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("                 Q-FORMAT / FIXED-POINT ANALYSIS")
    print("=" * 78)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading WM-811K...")

    df = load_wm811k(
        DATA_PATH
    )

    train_df = prepare_dataframe(
        df
    )

    print(
        "Calibration samples:",
        len(train_df)
    )

    dataset = WaferDataset(
        train_df,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WaferCNN(
        num_classes=9
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Activation ranges
    # --------------------------------------------------------

    print()
    print("Collecting activation ranges...")

    minimums, maximums = collect_activation_ranges(
        model,
        loader
    )

    # --------------------------------------------------------
    # Activation table
    # --------------------------------------------------------

    activation_types = {

        "features.0": "SIGNED CONV",
        "features.1": "UNSIGNED RELU",
        "features.2": "UNSIGNED POOL",

        "features.3": "SIGNED CONV",
        "features.4": "UNSIGNED RELU",
        "features.5": "UNSIGNED POOL",

        "features.6": "SIGNED CONV",
        "features.7": "UNSIGNED RELU",
        "features.8": "UNSIGNED POOL",

        "classifier.1": "SIGNED LINEAR",
        "classifier.2": "UNSIGNED RELU",
        "classifier.4": "SIGNED OUTPUT"
    }

    print()
    print("=" * 78)
    print("ACTIVATION Q-FORMAT ANALYSIS")
    print("=" * 78)

    print(
        f"{'Layer':<20}"
        f"{'Min':>12}"
        f"{'Max':>12}"
        f"{'Type':<18}"
    )

    print("-" * 78)

    for name, layer_type in activation_types.items():

        print(
            f"{name:<20}"
            f"{minimums[name]:>12.4f}"
            f"{maximums[name]:>12.4f}"
            f"{layer_type:<18}"
        )

    # --------------------------------------------------------
    # Candidate formats
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CANDIDATE FIXED-POINT FORMATS")
    print("=" * 78)

    candidate_bits = [
        8,
        7,
        6,
        5,
        4
    ]

    for bits in candidate_bits:

        print()
        print(
            f"----- TOTAL WIDTH: {bits}-BIT -----"
        )

        print(
            f"{'Layer':<20}"
            f"{'I':>4}"
            f"{'F':>4}"
            f"{'Min Rep':>14}"
            f"{'Max Rep':>14}"
            f"{'Fits':>8}"
        )

        print("-" * 78)

        for name, layer_type in activation_types.items():

            signed = (
                "SIGNED"
                in layer_type
            )

            integer_bits, fractional_bits, rep_min, rep_max, fits = (
                analyze_format(
                    minimums[name],
                    maximums[name],
                    bits,
                    signed
                )
            )

            print(
                f"{name:<20}"
                f"{integer_bits:>4}"
                f"{fractional_bits:>4}"
                f"{rep_min:>14.4f}"
                f"{rep_max:>14.4f}"
                f"{str(fits):>8}"
            )

    # --------------------------------------------------------
    # Weight analysis
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("WEIGHT RANGE ANALYSIS")
    print("=" * 78)

    weight_layers = {

        "features.0": model.features[0],
        "features.3": model.features[3],
        "features.6": model.features[6],
        "classifier.1": model.classifier[1],
        "classifier.4": model.classifier[4]
    }

    for name, layer in weight_layers.items():

        minimum, maximum = get_weight_range(
            layer.weight
        )

        print(
            f"{name:<20}"
            f"min={minimum:>10.6f}  "
            f"max={maximum:>10.6f}"
        )

    # --------------------------------------------------------
    # Accumulator estimation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("ACCUMULATOR RANGE ESTIMATION")
    print("=" * 78)

    print()
    print(
        "This estimates the worst-case MAC accumulation width."
    )

    conv_layers = {

        "Conv1": model.features[0],
        "Conv2": model.features[3],
        "Conv3": model.features[6]
    }

    for name, layer in conv_layers.items():

        kernel_elements = (
            layer.in_channels
            * layer.kernel_size[0]
            * layer.kernel_size[1]
        )

        weight_max = (
            layer.weight.detach()
            .abs()
            .max()
            .item()
        )

        print()
        print(
            f"{name}"
        )

        print(
            "  MAC terms per output:",
            kernel_elements
        )

        print(
            "  Maximum weight:",
            f"{weight_max:.6f}"
        )

        print(
            "  The accumulator must be wider "
            "than the operand widths."
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        "reports/q_format_analysis.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "Q-FORMAT / FIXED-POINT ANALYSIS\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Input size: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            f"Calibration samples: {len(train_df)}\n\n"
        )

        f.write(
            "ACTIVATION RANGES\n"
        )

        for name in activation_types:

            f.write(
                f"{name}: "
                f"{minimums[name]:.6f} "
                f"to "
                f"{maximums[name]:.6f}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "The candidate I.F formats above are "
            "range-based estimates.\n"
        )

        f.write(
            "They are NOT yet the final RTL fixed-point "
            "configuration.\n"
        )

    print()
    print("=" * 78)
    print("Q-FORMAT ANALYSIS COMPLETE")
    print("=" * 78)

    print()
    print(
        "Report saved to:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()
