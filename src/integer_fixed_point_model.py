import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from load_wm811k import load_wm811k
from model import WaferCNN


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/LSWMD.pkl"
MODEL_PATH = "models/baseline_cnn.pth"

IMAGE_SIZE = 48
BATCH_SIZE = 128

DEVICE = torch.device("cpu")


# Q formats
INPUT_FRAC = 0

CONV1_FRAC = 4
CONV2_FRAC = 4
CONV3_FRAC = 3

FC1_FRAC = 3
FC2_FRAC = 1

WEIGHT_FRAC = 6


# ============================================================
# Q FORMAT HELPERS
# ============================================================

def quantize_signed(x, frac_bits, total_bits):
    """
    Convert floating-point tensor to signed integer
    using fixed-point scaling.
    """

    scale = 2 ** frac_bits

    q = torch.round(x * scale)

    min_val = -(2 ** (total_bits - 1))
    max_val = (2 ** (total_bits - 1)) - 1

    q = torch.clamp(q, min_val, max_val)

    return q.to(torch.int64)


def quantize_unsigned(x, frac_bits, total_bits):
    """
    Convert floating-point tensor to unsigned integer.
    """

    scale = 2 ** frac_bits

    q = torch.round(x * scale)

    min_val = 0
    max_val = (2 ** total_bits) - 1

    q = torch.clamp(q, min_val, max_val)

    return q.to(torch.int64)


def dequantize(x, frac_bits):
    return x.float() / (2 ** frac_bits)


# ============================================================
# REQUANTIZATION
# ============================================================

def requantize(acc, acc_frac, output_frac, output_bits, signed=True):
    """
    Convert integer accumulator from one fixed-point scale
    to another fixed-point scale.
    """

    shift = acc_frac - output_frac

    if shift > 0:
        # Round before right shift
        rounding = 1 << (shift - 1)
        acc = torch.where(
            acc >= 0,
            acc + rounding,
            acc - rounding
        )

        acc = acc >> shift

    elif shift < 0:
        acc = acc << (-shift)

    if signed:
        min_val = -(2 ** (output_bits - 1))
        max_val = (2 ** (output_bits - 1)) - 1
    else:
        min_val = 0
        max_val = (2 ** output_bits) - 1

    acc = torch.clamp(acc, min_val, max_val)

    return acc


# ============================================================
# INTEGER CONVOLUTION
# ============================================================

def integer_conv2d(
    x,
    weight,
    bias,
    input_frac,
    weight_frac,
    output_frac,
    output_bits,
    stride=1,
    padding=1,
):
    """
    Integer convolution using unfold + integer matrix multiplication.

    x:
        [B, Cin, H, W]

    weight:
        [Cout, Cin, K, K]

    All arithmetic before requantization is integer.
    """

    B, Cin, H, W = x.shape
    Cout, _, K, _ = weight.shape

        # --------------------------------------------------------
    # INTEGER IM2COL
    # --------------------------------------------------------
    # torch.nn.functional.unfold() does not support int64
    # tensors on CPU, so construct the patches explicitly.
    #
    # Everything remains integer.
    # --------------------------------------------------------

    if padding > 0:
        x_padded = F.pad(
            x,
            (padding, padding, padding, padding),
            mode="constant",
            value=0
        )
    else:
        x_padded = x

    out_H = (
        (H + 2 * padding - K) // stride
    ) + 1

    out_W = (
        (W + 2 * padding - K) // stride
    ) + 1

    patch_list = []

    for row in range(out_H):

        row_start = row * stride
        row_end = row_start + K

        for col in range(out_W):

            col_start = col * stride
            col_end = col_start + K

            patch = x_padded[
                :,
                :,
                row_start:row_end,
                col_start:col_end
            ]

            patch = patch.reshape(
                B,
                -1
            )

            patch_list.append(patch)

    # [B, positions, Cin*K*K]
    patches = torch.stack(
        patch_list,
        dim=1
    )

    # [B, Cin*K*K, positions]
    patches = patches.transpose(
        1,
        2
    )

    # [Cout, Cin*K*K]
    weight_flat = weight.reshape(
        Cout,
        -1
    )

    outputs = []

    for b in range(B):

        # [Cout, positions]
        acc = torch.matmul(
            weight_flat,
            patches[b]
        )

        # Bias is represented at:
        #
        # input scale × weight scale
        #
        # Therefore bias integer uses:
        #
        # bias_float / (2^-input_frac × 2^-weight_frac)
        #
        # = bias_float × 2^(input_frac + weight_frac)

        if bias is not None:

            bias_int = torch.round(
                bias * (2 ** (input_frac + weight_frac))
            ).to(torch.int64)

            acc = acc + bias_int[:, None]

        # Accumulator fractional bits
        acc_frac = input_frac + weight_frac

        # Convert accumulator to output format
        out = requantize(
            acc,
            acc_frac,
            output_frac,
            output_bits,
            signed=True
        )

        outputs.append(out)

    out = torch.stack(outputs, dim=0)

    out_H = (H + 2 * padding - K) // stride + 1
    out_W = (W + 2 * padding - K) // stride + 1

    out = out.reshape(
        B,
        Cout,
        out_H,
        out_W
    )

    return out


# ============================================================
# INTEGER LINEAR
# ============================================================

def integer_linear(
    x,
    weight,
    bias,
    input_frac,
    weight_frac,
    output_frac,
    output_bits,
):
    """
    Integer fully-connected layer.
    """

    acc = torch.matmul(
        x,
        weight.t()
    )

    if bias is not None:

        bias_int = torch.round(
            bias * (2 ** (input_frac + weight_frac))
        ).to(torch.int64)

        acc = acc + bias_int

    acc_frac = input_frac + weight_frac

    out = requantize(
        acc,
        acc_frac,
        output_frac,
        output_bits,
        signed=True
    )

    return out


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("TRUE INTEGER FIXED-POINT GOLDEN MODEL")
print("=" * 70)

print("Device:", DEVICE)

print()
print("Loading WM-811K...")

df = load_wm811k(DATA_PATH)

print("Dataset:", df.shape)


# ============================================================
# LABEL EXTRACTION
# ============================================================

def extract_value(x):
    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


df["split"] = df["trianTestLabel"].apply(extract_value)
df["failure"] = df["failureType"].apply(extract_value)

train_df = df[df["split"] == "Training"].copy()
test_df = df[df["split"] == "Test"].copy()

print()
print("Training samples:", len(train_df))
print("Test samples    :", len(test_df))


# ============================================================
# DATA PREPARATION
# ============================================================

classes = [
    "none",
    "Center",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Scratch",
    "Random",
    "Donut",
    "Near-full",
]

class_to_idx = {
    name: idx
    for idx, name in enumerate(classes)
}


def prepare_dataframe(dataframe):

    images = []
    labels = []

    from PIL import Image

    for _, row in dataframe.iterrows():

        wafer = np.asarray(
            row["waferMap"],
            dtype=np.uint8
        )

        image = Image.fromarray(wafer)

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        image = np.asarray(
            image,
            dtype=np.int64
        )

        images.append(image)

        labels.append(
            class_to_idx[row["failure"]]
        )

    images = np.stack(images)

    labels = np.asarray(labels)

    return images, labels


print()
print("Preparing test data...")

test_images, test_labels = prepare_dataframe(
    test_df
)

print("Test image shape:", test_images.shape)


# ============================================================
# LOAD FP32 MODEL
# ============================================================

print()
print("Loading FP32 baseline...")

model = WaferCNN(
    num_classes=9
).to(DEVICE)

state = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state)

model.eval()


# ============================================================
# QUANTIZE MODEL WEIGHTS
# ============================================================

print()
print("=" * 70)
print("QUANTIZING WEIGHTS")
print("=" * 70)

q_weights = {}

for name, param in model.named_parameters():

    if "weight" in name:

        q_weights[name] = quantize_signed(
            param.data,
            WEIGHT_FRAC,
            total_bits=8
        )

        print(
            f"{name:25s}",
            "float range:",
            f"{param.data.min().item():.6f}",
            "to",
            f"{param.data.max().item():.6f}"
        )


# ============================================================
# INTEGER MODEL FORWARD
# ============================================================

def integer_forward(x_float):

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    x = quantize_unsigned(
        x_float,
        INPUT_FRAC,
        total_bits=2
    )

    # --------------------------------------------------------
    # CONV1
    # --------------------------------------------------------

    w = q_weights["features.0.weight"]

    b = model.features[0].bias.data

    x = integer_conv2d(
        x,
        w,
        b,
        input_frac=INPUT_FRAC,
        weight_frac=WEIGHT_FRAC,
        output_frac=CONV1_FRAC,
        output_bits=8
    )

    x = torch.relu(x)

    # --------------------------------------------------------
    # POOL1
    # --------------------------------------------------------

    x = F.max_pool2d(
        x,
        kernel_size=2
    )

    # --------------------------------------------------------
    # CONV2
    # --------------------------------------------------------

    w = q_weights["features.3.weight"]

    b = model.features[3].bias.data

    x = integer_conv2d(
        x,
        w,
        b,
        input_frac=CONV1_FRAC,
        weight_frac=WEIGHT_FRAC,
        output_frac=CONV2_FRAC,
        output_bits=8
    )

    x = torch.relu(x)

    # --------------------------------------------------------
    # POOL2
    # --------------------------------------------------------

    x = F.max_pool2d(
        x,
        kernel_size=2
    )

    # --------------------------------------------------------
    # CONV3
    # --------------------------------------------------------

    w = q_weights["features.6.weight"]

    b = model.features[6].bias.data

    x = integer_conv2d(
        x,
        w,
        b,
        input_frac=CONV2_FRAC,
        weight_frac=WEIGHT_FRAC,
        output_frac=CONV3_FRAC,
        output_bits=8
    )

    x = torch.relu(x)

    # --------------------------------------------------------
    # POOL3
    # --------------------------------------------------------

    x = F.max_pool2d(
        x,
        kernel_size=2
    )

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    x = x.reshape(
        x.shape[0],
        -1
    )

    # --------------------------------------------------------
    # FC1
    # --------------------------------------------------------

    w = q_weights["classifier.1.weight"]

    b = model.classifier[1].bias.data

    x = integer_linear(
        x,
        w,
        b,
        input_frac=CONV3_FRAC,
        weight_frac=WEIGHT_FRAC,
        output_frac=FC1_FRAC,
        output_bits=12
    )

    x = torch.relu(x)

    # --------------------------------------------------------
    # FC2
    # --------------------------------------------------------

    w = q_weights["classifier.4.weight"]

    b = model.classifier[4].bias.data

    x = integer_linear(
        x,
        w,
        b,
        input_frac=FC1_FRAC,
        weight_frac=WEIGHT_FRAC,
        output_frac=FC2_FRAC,
        output_bits=8
    )

    return x


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("INTEGER MODEL EVALUATION")
print("=" * 70)

all_predictions = []

num_samples = len(test_images)

with torch.no_grad():

    for start in tqdm(
        range(0, num_samples, BATCH_SIZE),
        desc="Testing"
    ):

        end = min(
            start + BATCH_SIZE,
            num_samples
        )

        batch = torch.from_numpy(
            test_images[start:end]
        ).unsqueeze(1).float()

        batch = batch.to(DEVICE)

        logits = integer_forward(
            batch
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    all_predictions
)

macro_f1 = f1_score(
    test_labels,
    all_predictions,
    average="macro"
)

print()
print("=" * 70)
print("TRUE INTEGER RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Macro F1 : {macro_f1:.6f}"
)

print()
print(
    classification_report(
        test_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)


# ============================================================
# SAVE REPORT
# ============================================================

report = classification_report(
    test_labels,
    all_predictions,
    target_names=classes,
    digits=6
)

with open(
    "reports/integer_fixed_point_results.txt",
    "w"
) as f:

    f.write(
        "TRUE INTEGER FIXED-POINT GOLDEN MODEL\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Configuration\n"
    )

    f.write(
        "Input  : unsigned Q2.0\n"
        "Conv1  : signed Q4.4\n"
        "Conv2  : signed Q4.4\n"
        "Conv3  : signed Q5.3\n"
        "FC1    : signed Q9.3 (12-bit)\n"
        "FC2    : signed Q7.1\n"
        "Weights: signed Q2.6 (8-bit)\n\n"
    )

    f.write(
        f"Accuracy : {accuracy:.6f}\n"
    )

    f.write(
        f"Macro F1 : {macro_f1:.6f}\n\n"
    )

    f.write(report)


print()
print(
    "Report saved to:"
)
print(
    "reports/integer_fixed_point_results.txt"
)
