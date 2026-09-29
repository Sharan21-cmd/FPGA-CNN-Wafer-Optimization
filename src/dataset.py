import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASSES = [
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

CLASS_TO_IDX = {
    name: idx for idx, name in enumerate(CLASSES)
}


class WaferDataset(Dataset):

    def __init__(self, dataframe, image_size=48):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        wafer = np.asarray(
            row["waferMap"],
            dtype=np.uint8
        )

        # Preserve categorical wafer-map values
        image = Image.fromarray(wafer)

        # Resize using nearest-neighbor
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = np.asarray(image, dtype=np.float32)

        # 1 × 48 × 48
        image = torch.from_numpy(image).unsqueeze(0)

        label = CLASS_TO_IDX[row["failure"]]

        return image, torch.tensor(label, dtype=torch.long)
