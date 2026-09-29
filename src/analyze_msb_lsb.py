import numpy as np
import pandas as pd
import pickle


DATA_PATH = "data/raw/LSWMD.pkl"


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
        df = CustomUnpickler(
            f,
            encoding="latin1"
        ).load()

    return df


def extract_label(x):

    try:
        return str(x.flat[0])
    except Exception:
        return "unknown"


def analyze_bits(df):

    df = df.copy()

    df["failure"] = df["failureType"].apply(
        extract_label
    )

    df["split"] = df["trianTestLabel"].apply(
        extract_label
    )

    valid = [
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
        df["failure"].isin(valid)
        & df["split"].isin(["Training", "Test"])
    ]

    print("\n======================================")
    print("       MSB / LSB BIT ANALYSIS")
    print("======================================")

    print("\nSamples analyzed:", len(df))

    # ------------------------------------------------------
    # Global value statistics
    # ------------------------------------------------------

    counts = {
        0: 0,
        1: 0,
        2: 0
    }

    for wafer in df["waferMap"]:

        arr = np.asarray(
            wafer,
            dtype=np.uint8
        )

        values, value_counts = np.unique(
            arr,
            return_counts=True
        )

        for value, count in zip(
            values,
            value_counts
        ):

            counts[int(value)] += int(count)

    total = sum(counts.values())

    print("\n===== VALUE DISTRIBUTION =====")

    for value in [0, 1, 2]:

        percentage = (
            counts[value] / total
        ) * 100

        print(
            f"Value {value}: "
            f"{counts[value]:,} "
            f"({percentage:.4f}%)"
        )

    # ------------------------------------------------------
    # Binary representation
    # ------------------------------------------------------

    print("\n===== 8-BIT REPRESENTATION =====")

    for value in [0, 1, 2]:

        binary = format(
            value,
            "08b"
        )

        print(
            f"{value} = {binary}"
        )

    # ------------------------------------------------------
    # Bit activity
    # ------------------------------------------------------

    bit_counts = np.zeros(
        8,
        dtype=np.int64
    )

    total_pixels = 0

    for wafer in df["waferMap"]:

        arr = np.asarray(
            wafer,
            dtype=np.uint8
        )

        flat = arr.reshape(-1)

        total_pixels += len(flat)

        for bit in range(8):

            bit_counts[bit] += np.count_nonzero(
                (flat >> bit) & 1
            )

    print("\n===== BIT ACTIVITY =====")

    print(
        "Bit numbering: "
        "bit 7 = MSB, bit 0 = LSB"
    )

    for bit in range(7, -1, -1):

        percentage = (
            bit_counts[bit]
            / total_pixels
        ) * 100

        name = "MSB" if bit == 7 else (
            "LSB" if bit == 0 else ""
        )

        print(
            f"Bit {bit}: "
            f"{bit_counts[bit]:,} "
            f"({percentage:.6f}%) "
            f"{name}"
        )

    # ------------------------------------------------------
    # Information from retained bits
    # ------------------------------------------------------

    print("\n===== RETAINED-BIT REPRESENTATIONS =====")

    configurations = {
        "Full 8-bit": list(range(8)),
        "Lower 4-bit (LSB)": list(range(4)),
        "Upper 4-bit (MSB)": list(range(4, 8)),
        "Lower 3-bit (LSB)": list(range(3)),
        "Upper 3-bit (MSB)": list(range(5, 8)),
        "Lower 2-bit (LSB)": list(range(2)),
        "Upper 2-bit (MSB)": list(range(6, 8)),
    }

    for name, bits in configurations.items():

        representations = {}

        for value in [0, 1, 2]:

            reduced = 0

            for new_bit, old_bit in enumerate(bits):

                if (value >> old_bit) & 1:

                    reduced |= (
                        1 << new_bit
                    )

            representations[value] = reduced

        print(
            f"{name:25s}: "
            f"{representations}"
        )

    # ------------------------------------------------------
    # Class-wise value distribution
    # ------------------------------------------------------

    print("\n===== CLASS-WISE VALUE DISTRIBUTION =====")

    for cls in valid:

        class_df = df[
            df["failure"] == cls
        ]

        class_counts = {
            0: 0,
            1: 0,
            2: 0
        }

        for wafer in class_df["waferMap"]:

            arr = np.asarray(
                wafer,
                dtype=np.uint8
            )

            values, value_counts = np.unique(
                arr,
                return_counts=True
            )

            for value, count in zip(
                values,
                value_counts
            ):

                class_counts[int(value)] += int(count)

        class_total = sum(
            class_counts.values()
        )

        if class_total == 0:
            continue

        print(
            f"\n{cls}:"
        )

        for value in [0, 1, 2]:

            percentage = (
                class_counts[value]
                / class_total
            ) * 100

            print(
                f"  Value {value}: "
                f"{percentage:.4f}%"
            )

    print("\n======================================")
    print("Analysis complete.")
    print("======================================")


if __name__ == "__main__":
    df = load_wm811k(DATA_PATH)
    analyze_bits(df)
