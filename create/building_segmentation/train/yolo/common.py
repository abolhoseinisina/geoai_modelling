import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
_BUILDING_SEG = HERE.parent.parent
if str(_BUILDING_SEG) not in sys.path:
    sys.path.insert(0, str(_BUILDING_SEG))

from nms import NMS_IOU, georeferencePolygons, performNMS

TILES_DIR = HERE / "tiles"
TILE_INDEX = TILES_DIR / "index.json"
DATASET_DIR = HERE / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
OUTPUT_DIR = HERE / "output"
WEIGHTS_PATH = OUTPUT_DIR / "train/weights/best.pt"
ONNX_PATH = OUTPUT_DIR / "train/weights/best.onnx"

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

def trainYOLOModel(yolo_base_model, data_yaml: Path, device, epochs: int, tile_size: int, batch_size: int, workers: int, patience: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(yolo_base_model)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=tile_size, batch=batch_size, device=device, workers=workers, patience=patience, project=str(output_dir), name="train", exist_ok=True, plots=True)
    return YOLO(str(output_dir / "train/weights/best.pt"))

def validateYOLOModel(model: YOLO, validation_tiles_dir: Path, validation_tile_index: dict, tile_size: int, score_threshold: float, truth_by_source: dict[str, list[Polygon]]):
    validation_df = pd.DataFrame(validation_tile_index)
    
    results = []
    for raster, tiles in tqdm(validation_df.groupby('source'), desc='Validate YOLO', ncols=100):
        polygons: list[Polygon] = []
        scores: list[float] = []
        for _, tile in tiles.iterrows():
            image_path = validation_tiles_dir / tile['image']
            rgb = np.array(Image.open(image_path).convert("RGB"))
            bgr = rgb[:, :, ::-1]
            result = model.predict(bgr, imgsz=tile_size, conf=0.01, verbose=False)[0]
            pixel_polys = []
            pixel_scores = []
            if result.masks is not None and result.boxes is not None:
                for xy, score in zip(result.masks.xy, result.boxes.conf.cpu().numpy()):
                    if float(score) < score_threshold or len(xy) < 3:
                        continue
                    pixel_polys.append(Polygon(xy))
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