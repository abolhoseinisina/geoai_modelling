import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import geopandas as gpd
from pathlib import Path
import onnxruntime as ort
from ultralytics import YOLO
import matplotlib.pyplot as plt
from dataclasses import dataclass
from shapely.geometry import Polygon

from device import getDevice
from train.yolo.utils import validateYOLOModel
from nms import performNMS, georeferencePolygon
from accuracy import getPrecisionRecall, getIoUDice
from tiling import generateTiles, loadGroundTruthBySource
from finetune.utils import validateMaskRCNNModel, convertMask2Polygonpx
from config import getFinalModelConfig, SEED, VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE

IOU_THRESH = 0.5
OVERVIEW_SIZE = 1400
OUT_DIR = "output/compare_models"
BLANK_FRACTION = 0.6

@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    model_type: str
    tile_size: int
    overlap: int
    gsd_m: float

MODELS = [
    ModelSpec("MASK-RCNN", Path("../../models/building_footprints_usa.pth"), "MASK-RCNN", tile_size=640, overlap=64, gsd_m=0.25),
    ModelSpec("FINE-TUNED MASK-RCNN", Path("output/models/finetune/finetuned_building_footprints_usa_1ep_20260911.pth"), "MASK-RCNN", tile_size=640, overlap=64, gsd_m=0.1),
    ModelSpec("FINE-TUNED MASK-RCNN (ONNX)", Path("models/finetuned_building_footprints_usa.onnx"), "MASK-RCNN-ONNX", tile_size=640, overlap=64, gsd_m=0.1),
    ModelSpec("YOLO", Path("output/models/train/weights/best.pt"), "YOLO", tile_size=640, overlap=64, gsd_m=0.1),
    ModelSpec("YOLO (ONNX)", Path("models/yolo_80ep.onnx"), "YOLO-ONNX", tile_size=512, overlap=64, gsd_m=0.1),
    ModelSpec("RAMP XUNET (ONNX)", Path("../../models/buildings_ramp_XUnet_256.onnx"), "XUNET-ONNX", tile_size=256, overlap=13, gsd_m=0.50),
]

def getEnvs(results: pd.DataFrame):
    envs = {
        'crop_0': 'mixed',
        'crop_1': 'tall',
        'crop_2': 'wide',
        'crop_3': 'residential',
    }

    def get_env(source: str):
        for name, env in envs.items():
            if name in source:
                return env
        
        return None

    results['env'] = results['source'].apply(get_env)
    return results

def plotHeatmapPerModelEnv(results, column_names: list[str], title: str):
    envs = sorted(results['env'].unique())
    models = sorted(results['model'].unique())

    cmap = plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    for idx, column_name in enumerate(column_names):
        heatmap_data = results.pivot_table(index='model', columns='env', values=column_name, aggfunc='mean')
        heatmap_data = heatmap_data.reindex(index=models, columns=envs)
        heatmap_values = heatmap_data.values
        ax = axes[idx]
        im = ax.imshow(heatmap_values, aspect="auto", cmap=cmap, vmin=np.nanmin(heatmap_values), vmax=np.nanmax(heatmap_values))
        ax.set_xticks(np.arange(len(envs)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(envs)
        ax.set_yticklabels(models if idx == 1 else [""]*len(models))
        ax.set_xlabel('Environment')
        if idx == 0:
            ax.set_ylabel('Model')
        
        for i in range(len(models)):
            for j in range(len(envs)):
                val = heatmap_values[i, j]
                text = f"{val:.2f}" if not np.isnan(val) else "NA"
                ax.text(j, i, text, ha="center", va="center", color="black" if np.isnan(val) or val < (np.nanmax(heatmap_values) * 0.7) else "white", fontsize=10)
        
        ax.set_title(column_name)
        if idx == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=title)
    
    plt.suptitle(f"{title} Heatmap")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'output/models/compare_per_model_per_env.jpg', dpi=300)
    plt.close()

def drawComparisonChart(file_path):
    comparison_results = pd.read_csv(file_path, index_col=0)
    results = getEnvs(comparison_results)
    
    results['f1'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
    results['f1'] = results['f1'].fillna(0)
    results['f1_weighted'] = results['f1'] * results['actual']
    
    plotHeatmapPerModelEnv(results, ['f1', 'dice', 'iou'], 'Compare Models')

def ortSession(path: Path):
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")

    return ort.InferenceSession(str(path), providers=providers)

def polygonsFromMask(mask: np.ndarray, min_area_px: float = 16) -> list[Polygon]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        eps = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) < 3:
            continue
        polygons.append(Polygon([(float(x), float(y)) for x, y in approx.reshape(-1, 2)]))
    return polygons

def validateMaskRCNNOnnx(session, validation_tiles_dir: Path, validation_tile_index: dict, tile_size: int, score_threshold: float, truth_by_source: dict[str, list[Polygon]], nms_iou_thresh: float, accuracy_iou_thresh: float):
    mask_threshold = 0.5
    input_name = session.get_inputs()[0].name
    validation_df = pd.DataFrame(validation_tile_index)
    results = []
    for raster, tiles in tqdm(validation_df.groupby('source'), desc='Validate MASK-RCNN-ONNX', ncols=100):
        polygons: list[Polygon] = []
        scores: list[float] = []
        for _, tile in tiles.iterrows():
            image_path = validation_tiles_dir / tile['image']
            rgb = np.array(Image.open(image_path).convert("RGB"))
            image = rgb.astype(np.float32) / 255.0   # (H, W, C)
            image = np.transpose(image, (2, 0, 1))   # (C, H, W)
            boxes, _labels, det_scores, masks = session.run(None, {input_name: image})
            
            if len(det_scores) == 0:
                continue
   
            for mask, score, box in zip(masks, det_scores, boxes):
                if score < score_threshold:
                    continue
            
                mask_2d = mask[0] if mask.ndim == 3 else mask
                binary = mask_2d > mask_threshold
                x1, y1, x2, y2 = [int(v) for v in box]
                cropped = np.zeros_like(binary, dtype=np.uint8)
                y1c, y2c = max(y1, 0), min(y2, binary.shape[0])
                x1c, x2c = max(x1, 0), min(x2, binary.shape[1])
                cropped[y1c:y2c, x1c:x2c] = binary[y1c:y2c, x1c:x2c]
                ring = convertMask2Polygonpx(cropped)
                if ring is None:
                    continue
                
                poly_map = georeferencePolygon(Polygon(ring), tile['transform'])
                if poly_map is not None:
                    polygons.append(poly_map)
                    scores.append(float(score))

        keep = performNMS(polygons, scores, nms_iou_thresh)
        predicted = [polygons[i] for i in keep]
        ground_truth = truth_by_source.get(raster, [])
        recall, precision, true_positives, false_positives = getPrecisionRecall(predicted, ground_truth, accuracy_iou_thresh)
        iou, dice = getIoUDice(predicted, ground_truth)
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
    
def buildingProbability(raw: np.ndarray) -> np.ndarray:
    pred = np.squeeze(raw).astype(np.float32)
    if pred.ndim == 3:
        if pred.shape[-1] <= 4 and pred.shape[0] > 4:
            pred = np.moveaxis(pred, -1, 0)

        if pred.shape[0] == 1:
            pred = pred[0]
        
        else:
            shifted = pred - pred.max(axis=0, keepdims=True)
            exponentiated = np.exp(shifted)
            pred = (exponentiated / exponentiated.sum(axis=0, keepdims=True))[1]

    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = 1.0 / (1.0 + np.exp(-pred))

    return pred

def validateXunetOnnx(session, validation_tiles_dir: Path, validation_tile_index: dict, tile_size: int, gsd_m: float, truth_by_source: dict[str, list[Polygon]], nms_iou_thresh: float, accuracy_iou_thresh: float) -> gpd.GeoDataFrame:
    peak = 0.0
    seg_thresh = 0.5
    min_segment_px = 11
    
    input = session.get_inputs()[0]
    input_name = input.name
    validation_df = pd.DataFrame(validation_tile_index)
    results = []
    for raster, tiles in tqdm(validation_df.groupby('source'), desc='Validate MASK-RCNN-ONNX', ncols=100):
        polygons: list[Polygon] = []
        scores: list[float] = []
        for _, tile in tiles.iterrows():
            image_path = validation_tiles_dir / tile['image']
            rgb = np.array(Image.open(image_path).convert("RGB"))  # shape (H, W, C)
            image = rgb.astype(np.float32) / 255.0  # normalize to 0-1, shape (H, W, C)
            if image.shape[-1] == 3:  # channel last
                image = np.transpose(image, (2, 0, 1))  # make it (C, H, W)
            
            image = image[None, ...]  # add batch dimension (1, C, H, W)
            pred = buildingProbability(session.run(None, {input_name: image})[0])
            peak = max(peak, float(pred.max()))
            binary = (pred > seg_thresh).astype(np.uint8)
            if not binary.any():
                continue
            
            for poly_px in polygonsFromMask(binary, min_segment_px):
                poly_map = georeferencePolygon(poly_px, tile['transform'])
                if poly_map is not None:
                    polygons.append(poly_map)
                    scores.append(1.0)

        keep = performNMS(polygons, scores, nms_iou_thresh)
        predicted = [polygons[i] for i in keep]
        ground_truth = truth_by_source.get(raster, [])
        recall, precision, true_positives, false_positives = getPrecisionRecall(predicted, ground_truth, accuracy_iou_thresh)
        iou, dice = getIoUDice(predicted, ground_truth)
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

def validate(model_spec: ModelSpec, validation_tile_dir: str, validation_tile_index: dict, truth_by_source, device, nms_iou_threshold, accuracy_iou_threshold):
    if model_spec.model_type == "MASK-RCNN":
        return validateMaskRCNNModel(model_spec.path, validation_tile_dir, validation_tile_index, device, 0.5, model_spec.tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)
    
    elif model_spec.model_type == "MASK-RCNN-ONNX":
        session = ortSession(model_spec.path)
        return validateMaskRCNNOnnx(session, validation_tile_dir, validation_tile_index, model_spec.tile_size, 0.5, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)
    
    elif model_spec.model_type == "XUNET-ONNX":
        session = ortSession(model_spec.path)
        return validateXunetOnnx(session, validation_tile_dir, validation_tile_index, model_spec.tile_size, model_spec.gsd_m, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)
    
    elif model_spec.model_type == "YOLO-ONNX":
        model = YOLO(model_spec.path, task='segment', verbose=False)
        return validateYOLOModel(model, validation_tile_dir, validation_tile_index, model_spec.tile_size, 0.5, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)
    
    elif model_spec.model_type == "YOLO":
        model = YOLO(model_spec.path, task='segment')
        return validateYOLOModel(model, validation_tile_dir, validation_tile_index, model_spec.tile_size, 0.5, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)
    
    else:
        raise SystemExit(f"unknown model kind: {model_spec.kind}")

def execute() -> None:
    config = getFinalModelConfig()
    
    available_models = [spec for spec in MODELS if spec.path.exists()]
    for spec in MODELS:
        if not spec.path.exists():
            print(f"Warning: Skipping {spec.name}. Path does not exist: {spec.path}")
    
    if not available_models:
        raise SystemExit("Error: No model files found.")

    device = getDevice("auto")
    print(f"Device: {device}")
    
    truth_by_source = loadGroundTruthBySource(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE)
    results = []
    for model_spec in available_models:
        validation_tile_index = generateTiles(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE, model_spec.tile_size, model_spec.overlap, gsd_m=model_spec.gsd_m, seed=SEED)
        model_results = validate(model_spec, VALIDATING_TILES_DIR, validation_tile_index, truth_by_source, device, config['nms_iou_threshold'], config['accuracy_iou_threshold'])
        for row in model_results:
            row['model'] = model_spec.name
            results.append(row)
    
    results = pd.DataFrame(results, columns=["model", "source", "actual", "predicted", "recall", "precision", "true_positives", "false_positives", "iou", "dice"])
    results.to_csv('output/models/comparison.csv')
    drawComparisonChart('output/models/comparison.csv')

if __name__ == "__main__":
    execute()