import cv2
import sys
import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from shapely.ops import unary_union
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from shapely.validation import make_valid
from .BuildingTileDataset import BuildingTileDataset
from torchvision.models.detection import maskrcnn_resnet50_fpn

HERE = Path(__file__).resolve().parent
_BUILDING_SEG = HERE.parent
if str(_BUILDING_SEG) not in sys.path:
    sys.path.insert(0, str(_BUILDING_SEG))

from device import cudaFreeBytes, getDevice, pickCudaDevice  # noqa: E402, F401
from geo import BUILDINGS_GEOJSON, DATA_DIR, TILE_SIZE, getRasterResolution, getSamplingScales, getWindowOrigins  # noqa: E402
from nms import NMS_IOU, georeferencePolygons, performNMS  # noqa: E402

PRETRAINED_PATH = HERE.parent.parent.parent / "models/building_footprints_usa.pth"
FINETUNED_PATH = HERE / "output/finetuned_building_footprints_usa.pth"
ONNX_PATH = HERE / "output/finetuned_building_footprints_usa.onnx"
TILES_DIR = HERE / "tiles"
TILE_INDEX = TILES_DIR / "index.json"
NUM_CLASSES = 2  # 0 = background, 1 = building
_BACKBONE_STAGES = ("layer4", "layer3", "layer2", "layer1", "conv1")

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 10.0
TRAINABLE_BACKBONE_STAGES = 3

IOU_THRESH = 0.5

def _safePolygon(poly: Polygon) -> Polygon | None:
    if poly.is_empty or len(poly.exterior.coords) < 4:
        return None

    fixed = make_valid(poly)
    if fixed.geom_type == "Polygon" and not fixed.is_empty:
        return fixed
    
    if fixed.geom_type == "MultiPolygon":
        return max(fixed.geoms, key=lambda g: g.area)
    
    return None

def _safePolygons(polys: list[Polygon]) -> list[Polygon]:
    out = []
    for poly in polys:
        fixed = _safePolygon(poly)
        if fixed is not None:
            out.append(fixed)
    
    return out

def _getPolygonIoU(a: Polygon, b: Polygon) -> float:
    a, b = _safePolygon(a), _safePolygon(b)
    if a is None or b is None:
        return 0.0

    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0

def _getPrecisionRecall(predicted: list[Polygon], truth: list[Polygon], iou_thresh: float = IOU_THRESH) -> tuple[float, float, int, int]:
    used = set()
    true_positives = 0
    for truth_poly in truth:
        best_i, best_iou = -1, 0.0
        for i, pred_poly in enumerate(predicted):
            if i in used:
                continue
            
            iou = _getPolygonIoU(truth_poly, pred_poly)
            if iou > best_iou:
                best_iou, best_i = iou, i
        
        if best_iou >= iou_thresh:
            true_positives += 1
            used.add(best_i)
    
    false_positives = max(len(predicted) - true_positives, 0)
    recall = true_positives / max(len(truth), 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    return recall, precision, true_positives, false_positives

def _getIoUDice(predicted: list[Polygon], truth: list[Polygon]) -> tuple[float, float]:
    predicted = _safePolygons(predicted)
    truth = _safePolygons(truth)

    if not predicted and not truth:
        return 1.0, 1.0
    
    if not predicted or not truth:
        return 0.0, 0.0

    pred_union = unary_union(predicted)
    truth_union = unary_union(truth)
    if pred_union.is_empty or truth_union.is_empty:
        return 0.0, 0.0

    intersection = pred_union.intersection(truth_union).area
    union = pred_union.union(truth_union).area
    iou = intersection / union if union > 0 else 0.0
    denom = pred_union.area + truth_union.area
    dice = (2.0 * intersection / denom) if denom > 0 else 0.0
    return iou, dice

def loadAndUnwrapStateDict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def buildModel(weights_path: Path, num_classes: int, image_size: int) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes, min_size=image_size, max_size=image_size)
    if weights_path is not None:
        model.load_state_dict(loadAndUnwrapStateDict(weights_path))

    return model

def freezeBackbone(model: nn.Module, trainable_stages: int) -> None:
    keep = _BACKBONE_STAGES[:trainable_stages]
    for name, param in model.backbone.body.named_parameters():
        if not any(name.startswith(stage) for stage in keep):
            param.requires_grad_(False)

def freezeBatchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)


def _splitTrainingValidationRecords(records: list[dict], validation_fraction: float, random_seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(random_seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * validation_fraction))
    return shuffled[cut:], shuffled[:cut]

def _collate(batch):
    return tuple(zip(*batch))

def _getLRSchedule(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return 0.05 + 0.95 * step / max(warmup_steps, 1)
        
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor

@torch.no_grad()
def _getValidationLoss(model, loader, device) -> dict[str, float]:
    model.train()
    freezeBatchnorm(model)
    totals: dict[str, float] = {}
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in t.items()} for t in targets]
        for key, value in model(images, targets).items():
            totals[key] = totals.get(key, 0.0) + value.item()
    
    return {key: value / max(len(loader), 1) for key, value in totals.items()}

def _convertMask2Polygonpx(mask: np.ndarray) -> list[tuple[float, float]] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 16:
        return None
    
    eps = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    if len(approx) < 3:
        return None
    
    return [(float(x), float(y)) for x, y in approx.reshape(-1, 2)]

@torch.no_grad()
def _applyModel2Tile(model, image_chw: torch.Tensor, device, score_threshold) -> list[tuple[Polygon, float]]:
    out = model([image_chw.to(device)])[0]
    keep = out["scores"] >= score_threshold
    if not keep.any():
        return []

    boxes = out["boxes"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    masks = out["masks"][keep, 0].cpu().numpy() > score_threshold
    results = []
    for mask, score, box in zip(masks, scores, boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cropped = np.zeros_like(mask, dtype=np.uint8)
        y1, y2 = max(y1, 0), min(y2, mask.shape[0])
        x1, x2 = max(x1, 0), min(x2, mask.shape[1])
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        ring = _convertMask2Polygonpx(cropped)
        if ring is None:
            continue

        results.append((Polygon(ring), float(score)))
    
    return results

def finetuneMaskRCNN(pretrained_model_path: str, tile_index: list[dict], tiles_dir: Path, tile_size: int, random_seed: int, validation_fraction: float, batch_size: int, num_workers: int, pin_memory: bool, device, epochs, learning_rate, output_models_dir):
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    train_records, val_records = _splitTrainingValidationRecords(tile_index, validation_fraction, random_seed)
    loader_kwargs = dict(batch_size=batch_size, collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(BuildingTileDataset(tiles_dir, train_records, augment=True), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(BuildingTileDataset(tiles_dir, val_records, augment=False), shuffle=False, **loader_kwargs)

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.backends.cudnn.benchmark = True

    model = buildModel(weights_path=pretrained_model_path, num_classes=2, image_size=tile_size)
    freezeBackbone(model, TRAINABLE_BACKBONE_STAGES)
    freezeBatchnorm(model)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _getLRSchedule(total_steps, warmup_steps=min(len(train_loader), 100)))
    
    output_model_path = output_models_dir / 'finetune/finetuned_building_footprints_usa.pth'
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    epoch_tqdm = tqdm(range(epochs), desc='Fine-tuning', ncols=100)
    for _ in epoch_tqdm:
        model.train()
        freezeBatchnorm(model)

        running, batches = 0.0, 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("  skipping batch with non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            batches += 1

        train_loss = running / max(batches, 1)
        val_losses = _getValidationLoss(model, val_loader, device)
        val_loss = sum(val_losses.values())
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_model_path)
        
        epoch_tqdm.set_postfix_str(f"train {train_loss:.4f}  val {val_loss:.4f}")

    model.load_state_dict(torch.load(output_model_path, map_location=device, weights_only=True))
    return output_model_path

def validateMaskRCNNModel(model, validation_tiles_dir, validation_tile_index, device, score_threshold, tile_size: int, truth_by_source: dict[str, list[Polygon]]):
    model = buildModel(weights_path=model, num_classes=2, image_size=tile_size)
    model.to(device).eval()
    
    validation_df = pd.DataFrame(validation_tile_index)
    results = []
    for raster, tiles in tqdm(validation_df.groupby('source'), desc='Validate Finetune', ncols=100):
        polygons: list[Polygon] = []
        scores: list[float] = []
        for _, tile in tiles.iterrows():
            image_path = validation_tiles_dir / tile['image']
            rgb = np.array(Image.open(image_path).convert("RGB"))
            chw = np.transpose(rgb, (2, 0, 1))
            tensor = torch.from_numpy(chw).float().div_(255)
            result = _applyModel2Tile(model, tensor, device, score_threshold)
            pixel_polys = []
            pixel_scores = []
            for poly_px, score in result:
                if float(score) < score_threshold:
                    continue
                pixel_polys.append(poly_px)
                pixel_scores.append(float(score))

            mapped, mapped_scores = georeferencePolygons(pixel_polys, pixel_scores, tile['transform'])
            polygons.extend(mapped)
            scores.extend(mapped_scores)

        keep = performNMS(polygons, scores, NMS_IOU)
        predicted = [polygons[i] for i in keep]
        ground_truth = truth_by_source.get(raster, [])
        recall, precision, true_positives, false_positives = _getPrecisionRecall(predicted, ground_truth)
        iou, dice = _getIoUDice(predicted, ground_truth)
        results.append({
            'source': raster,
            'actual': len(ground_truth),
            'predicted': len(predicted),
            'recall': recall,
            'precision': precision,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'iou': iou,
            'dice': dice,
        })
    
    return results