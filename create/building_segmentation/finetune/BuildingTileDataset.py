import torch
import random
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from rasterio.features import rasterize

class BuildingTileDataset(Dataset):
    """Tiles plus per-building instance masks, rasterized on demand."""

    def __init__(self, tiles_dir: str, records: list[dict], augment: bool):
        self.tiles_dir = tiles_dir
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _masks(self, record: dict) -> np.ndarray:
        size = record["size"]
        shape = (size, size)
        masks = [rasterize([(Polygon(rings[0], rings[1:]), 1)], out_shape=shape, dtype="uint8") for rings in record["polygons"]]
        if not masks:
            return np.zeros((0, size, size), dtype=np.uint8)

        return np.stack(masks)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = np.array(Image.open(self.tiles_dir / record["image"]).convert("RGB"))
        masks = self._masks(record)

        if self.augment:
            if random.random() < 0.5:
                image, masks = image[:, ::-1], masks[:, :, ::-1]
            
            if random.random() < 0.5:
                image, masks = image[::-1], masks[:, ::-1]
            
            turns = random.randint(0, 3)
            if turns:
                image = np.rot90(image, turns, axes=(0, 1))
                masks = np.rot90(masks, turns, axes=(1, 2))
            
            image, masks = np.ascontiguousarray(image), np.ascontiguousarray(masks)

        boxes, kept = [], []
        for position, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue

            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue

            boxes.append([x0, y0, x1, y1])
            kept.append(position)

        masks = masks[kept] if kept else np.zeros((0, *image.shape[:2]), dtype=np.uint8)
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)

        image_tensor = torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float().div_(255)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": torch.from_numpy(masks),
            "image_id": torch.tensor([index]),
            "area": torch.from_numpy((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image_tensor, target