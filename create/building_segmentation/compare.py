import sys
import cv2
import random
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
import onnxruntime as ort
from ultralytics import YOLO
from PIL import Image, ImageDraw
from dataclasses import dataclass
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FINE_TUNE = HERE / "fine_tune"
if str(FINE_TUNE) not in sys.path:
    sys.path.insert(0, str(FINE_TUNE))

from geo import MIN_GSD_M, TILE_SIZE, getSamplingScales, getWindowOrigins
from device import getDevice
from tiling import getRGBStretchBounds, loadTrainingBuildings, pairImagesWithFootprints, stretch2uint8, transformFootprints
from common import buildModel
from compare_models import MASK_THRESH, MAX_AREA_M2, MIN_AREA_M2, NMS_IOU, OVERLAP, SCORE_THRESH, applyModel2Raster, convertMask2Polygonpx, convertPixelToPolygon, getPolygonIoU, performNMS

IOU_THRESH = 0.5
OVERVIEW_SIZE = 1400
OUT_DIR = HERE / "output/compare"
BLANK_FRACTION = 0.6
REPO = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO / "validation_dataset/building_segmentation_202608"
BUILDINGS_GEOJSON = DATA_DIR / "validation_buildings.geojson"

# DATA_DIR = REPO / "training_dataset/building_segmentation_202608"
# BUILDINGS_GEOJSON = DATA_DIR / "training_buildings.geojson"

@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    kind: str
    tile_size: int = TILE_SIZE
    gsd_m: float = MIN_GSD_M
    overlap_px: int = OVERLAP
    seg_thresh: float = MASK_THRESH
    min_segment_px: int = 11


MODELS = [
    ModelSpec("pretrained_maskrcnn", HERE.parent.parent / "models/building_footprints_usa.pth", "maskrcnn_pth", gsd_m=0.25),
    ModelSpec("finetuned_maskrcnn_20ep", HERE / "models/finetuned_building_footprints_usa.onnx", "maskrcnn_onnx"),
    ModelSpec("yolo_80ep", HERE / "models/yolo_80ep.onnx", "yolo_onnx"),
    ModelSpec("ramp_xunet", HERE.parent.parent / "models/buildings_ramp_XUnet_256.onnx", "xunet_onnx", tile_size=256, gsd_m=0.50, overlap_px=13, seg_thresh=0.5, min_segment_px=11),
]


def ortSession(path: Path):
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(path), providers=providers)


def iterateTiles(src, tile_size: int, gsd_m: float = MIN_GSD_M, overlap_px: int = OVERLAP):
    scale_x, scale_y, target_gsd = getSamplingScales(src, gsd_m)
    source_width = tile_size * scale_x
    source_height = tile_size * scale_y
    band_indexes = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
    stretch = getRGBStretchBounds(src, band_indexes)
    xs = getWindowOrigins(src.width, source_width, (tile_size - overlap_px) * scale_x)
    ys = getWindowOrigins(src.height, source_height, (tile_size - overlap_px) * scale_y)
    for y0 in ys:
        for x0 in xs:
            window = Window(x0, y0, source_width, source_height)
            transform = src.window_transform(window) * Affine.scale(scale_x, scale_y)
            array = src.read(band_indexes, window=window, out_shape=(3, tile_size, tile_size), resampling=Resampling.bilinear, boundless=True, fill_value=0)
            if stretch is not None:
                array = stretch2uint8(array, *stretch)
            if (array.max(axis=0) == 0).mean() > BLANK_FRACTION:
                continue
            yield array, transform, target_gsd


def georeferencePolygon(poly_px: Polygon, transform, target_gsd: float) -> Polygon | None:
    """Filter a tile-pixel polygon by ground area, then project it into the raster CRS.

    The area test has to happen in pixel space: the rasters are EPSG:4326, so
    polygon area in map units is degrees squared, not metres squared.
    """
    if poly_px is None or poly_px.is_empty:
        return None

    area_m2 = poly_px.area * target_gsd**2
    if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
        return None

    poly_map = convertPixelToPolygon(poly_px, transform)
    if poly_map.is_empty or not poly_map.is_valid:
        return None

    return poly_map


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


def detectionsToFrame(polygons: list[Polygon], scores: list[float], crs) -> gpd.GeoDataFrame:
    if not polygons:
        return gpd.GeoDataFrame({"score": []}, geometry=[], crs=crs)
    
    keep = performNMS(polygons, scores, NMS_IOU)
    return gpd.GeoDataFrame({"score": [scores[i] for i in keep]}, geometry=[polygons[i] for i in keep], crs=crs)


def applyMaskRcnnOnnx(session, src, verbose: bool = True) -> gpd.GeoDataFrame:
    input_name = session.get_inputs()[0].name
    polygons: list[Polygon] = []
    scores: list[float] = []
    raw_detections = 0
    for array, transform, target_gsd in iterateTiles(src, TILE_SIZE, gsd_m=0.1):
        image = array.astype(np.float32) / 255.0
        boxes, _labels, det_scores, masks = session.run(None, {input_name: image})
        if len(det_scores) == 0:
            continue
        keep = det_scores >= SCORE_THRESH
        raw_detections += int(keep.sum())
        for mask, score, box in zip(masks[keep], det_scores[keep], boxes[keep]):
            mask_2d = mask[0] if mask.ndim == 3 else mask
            binary = mask_2d > MASK_THRESH
            x1, y1, x2, y2 = [int(v) for v in box]
            cropped = np.zeros_like(binary, dtype=np.uint8)
            y1c, y2c = max(y1, 0), min(y2, binary.shape[0])
            x1c, x2c = max(x1, 0), min(x2, binary.shape[1])
            cropped[y1c:y2c, x1c:x2c] = binary[y1c:y2c, x1c:x2c]
            ring = convertMask2Polygonpx(cropped)
            if ring is None:
                continue
            
            poly_map = georeferencePolygon(Polygon(ring), transform, target_gsd)
            if poly_map is not None:
                polygons.append(poly_map)
                scores.append(float(score))
    
    if verbose: print(f"  raw detections: {raw_detections}  kept after area filter: {len(polygons)}")
    return detectionsToFrame(polygons, scores, src.crs)


def buildingProbability(raw: np.ndarray) -> np.ndarray:
    """Reduce a semantic-segmentation output to a single building probability map."""
    pred = np.squeeze(raw).astype(np.float32)
    if pred.ndim == 3:
        # Channel-last outputs come back as (H, W, C); move the class axis first.
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


def applyXunetOnnx(session, src, spec: "ModelSpec", verbose: bool = True) -> gpd.GeoDataFrame:
    inp = session.get_inputs()[0]
    input_name = inp.name
    tile_size = spec.tile_size
    dims = [d if isinstance(d, int) and d > 0 else None for d in inp.shape]
    if len(dims) == 4 and dims[2]:
        tile_size = dims[2]

    polygons: list[Polygon] = []
    scores: list[float] = []
    positive_tiles = 0
    peak = 0.0
    target_gsd = spec.gsd_m
    for array, transform, target_gsd in iterateTiles(src, tile_size, spec.gsd_m, spec.overlap_px):
        image = array.astype(np.float32) / 255.0
        if len(inp.shape) == 4:
            image = image[None, ...]

        pred = buildingProbability(session.run(None, {input_name: image})[0])
        peak = max(peak, float(pred.max()))
        binary = (pred > spec.seg_thresh).astype(np.uint8)
        if not binary.any():
            continue

        positive_tiles += 1
        for poly_px in polygonsFromMask(binary, spec.min_segment_px):
            poly_map = georeferencePolygon(poly_px, transform, target_gsd)
            if poly_map is not None:
                polygons.append(poly_map)
                scores.append(1.0)

    if verbose: print(f"  tiles with mask: {positive_tiles}  peak probability: {peak:.3f}  gsd={target_gsd * 100:.0f} cm/px  tile={tile_size}")
    return detectionsToFrame(polygons, scores, src.crs)


def applyYoloOnnx(path: Path, src, verbose: bool = True) -> gpd.GeoDataFrame:
    model = YOLO(str(path), task='segment')
    polygons: list[Polygon] = []
    scores: list[float] = []
    raw_detections = 0
    peak = 0.0
    # Predict well below SCORE_THRESH so we can report the peak confidence even
    # when nothing clears the bar, which distinguishes a weak model from a bug.
    for array, transform, target_gsd in iterateTiles(src, TILE_SIZE, gsd_m=0.1, overlap_px=192):
        # Ultralytics reads numpy inputs as BGR, so hand it BGR rather than RGB.
        bgr = np.moveaxis(array, 0, -1)[:, :, ::-1]
        result = model.predict(bgr, imgsz=TILE_SIZE, conf=0.01, verbose=False)[0]
        if result.masks is None or result.boxes is None:
            continue
        
        for xy, score in zip(result.masks.xy, result.boxes.conf.cpu().numpy()):
            peak = max(peak, float(score))
            if score < SCORE_THRESH or len(xy) < 3:
                continue
            
            raw_detections += 1
            poly_map = georeferencePolygon(Polygon(xy), transform, target_gsd)
            if poly_map is not None:
                polygons.append(poly_map)
                scores.append(float(score))
    
    if verbose: print(f"  raw detections: {raw_detections}  kept after area filter: {len(polygons)}  peak confidence: {peak:.3f}")
    if peak < SCORE_THRESH:
        print(f"  no detection reached score>={SCORE_THRESH}: these weights are undertrained")
    return detectionsToFrame(polygons, scores, src.crs)

def groundTruthPolygons(src, footprints: gpd.GeoDataFrame) -> list[Polygon]:
    gdf = transformFootprints(footprints, src)
    raster_box = box(*src.bounds)
    polygons = []
    for geom in gdf.geometry:
        clipped = geom.intersection(raster_box)
        if clipped.is_empty:
            continue
        parts = [clipped] if clipped.geom_type == "Polygon" else list(getattr(clipped, "geoms", []))
        for part in parts:
            if part.geom_type == "Polygon" and part.is_valid and not part.is_empty:
                polygons.append(part)
    return polygons


def precisionRecall(predicted: list[Polygon], truth: list[Polygon], iou_thresh: float = IOU_THRESH) -> tuple[float, float, int, int]:
    used = set()
    true_positives = 0
    for truth_poly in truth:
        best_i, best_iou = -1, 0.0
        for i, pred_poly in enumerate(predicted):
            if i in used:
                continue
            iou = getPolygonIoU(truth_poly, pred_poly)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou >= iou_thresh:
            true_positives += 1
            used.add(best_i)
    false_positives = max(len(predicted) - true_positives, 0)
    recall = true_positives / max(len(truth), 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    return recall, precision, true_positives, false_positives


def iouDice(predicted: list[Polygon], truth: list[Polygon]) -> tuple[float, float]:
    """Image-level IoU and Dice from the union of predicted vs ground-truth polygons."""
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


def drawOverview(src, truth: list[Polygon], frames: dict[str, gpd.GeoDataFrame], metrics: dict, out_path: Path) -> None:
    scale = max(src.width, src.height) / OVERVIEW_SIZE
    out_w = max(1, int(round(src.width / scale)))
    out_h = max(1, int(round(src.height / scale)))
    rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
    base = np.moveaxis(rgb, 0, -1)
    overview_transform = src.transform * rasterio.Affine.scale(scale)
    inv = ~overview_transform

    def drawPolys(panel, geoms, color):
        draw = ImageDraw.Draw(panel, "RGBA")
        fill = (*color, 70)
        outline = (*color, 220)
        for geom in geoms:
            if geom is None or geom.is_empty:
                continue
            parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for part in parts:
                if part.geom_type != "Polygon":
                    continue
                pixels = [(inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f) for x, y in part.exterior.coords]
                draw.polygon(pixels, fill=fill, outline=outline)

    names = ["ground_truth", *frames]
    colors = {
        "ground_truth": (255, 220, 40),
        "pretrained_maskrcnn": (255, 40, 40),
        "ramp_xunet": (255, 40, 40),
        "finetuned_maskrcnn_20ep": (255, 40, 40),
        "yolo_80ep": (255, 40, 40),
    }
    panels = []
    for name in names:
        panel = Image.fromarray(base.copy())
        geoms = truth if name == "ground_truth" else list(frames[name].geometry)
        color = colors.get(name, (255, 255, 255))
        drawPolys(panel, geoms, color)
        label = f"{name}  n={len(geoms)}"
        if name in metrics:
            recall, precision, _, _ = metrics[name]
            label += f"  R={recall:.3f}  P={precision:.3f}"
        draw = ImageDraw.Draw(panel, "RGBA")
        draw.rectangle([0, 0, out_w - 1, 28], fill=(0, 0, 0, 160))
        from PIL import ImageFont

        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except OSError:
            font = ImageFont.load_default(size=40)

        draw.text((8, 6), label, fill=(255, 255, 255, 255), font=font)
        panels.append(panel)

    # Build a canvas with 3 rows and 2 panels in each row
    n_cols = 2
    n_rows = 3
    padding = 8

    canvas_width = (out_w * n_cols) + padding * (n_cols - 1)
    canvas_height = (out_h * n_rows) + padding * (n_rows - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (20, 20, 20))

    for idx, panel in enumerate(panels):
        row = idx // n_cols
        col = idx % n_cols
        if row >= n_rows:
            break  # don't draw more than fits the rows*cols limit
        x = col * (out_w + padding)
        y = row * (out_h + padding)
        canvas.paste(panel, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    print(f"wrote {out_path}")


def runModel(spec: ModelSpec, src, device, verbose: bool = True) -> gpd.GeoDataFrame:
    if verbose: print(f"\n=== {spec.name}: {spec.path.name} ({spec.kind}) ===")
    if spec.kind == "maskrcnn_pth":
        model = buildModel(weights_path=spec.path)
        model.to(device).eval()
        gdf = applyModel2Raster(model, src, device)
        del model
        return gdf
    
    elif spec.kind == "maskrcnn_onnx":
        return applyMaskRcnnOnnx(ortSession(spec.path), src, verbose)
    
    elif spec.kind == "xunet_onnx":
        return applyXunetOnnx(ortSession(spec.path), src, spec, verbose)
    
    elif spec.kind == "yolo_onnx":
        return applyYoloOnnx(spec.path, src, verbose)
    
    raise SystemExit(f"unknown model kind: {spec.kind}")

def getGSDs(pairs):
    gsds = {}
    for tif_path, _ in pairs:
        with rasterio.open(tif_path) as src:
            res_x, res_y = src.res
            if src.crs and src.crs.is_projected:
                gsd = abs(res_y)
            else:
                from rasterio.warp import calculate_default_transform
                transform, width, height = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
                gsd = abs(transform.e)
   
            gsds[tif_path.stem] = round(gsd, 3)
    
    return gsds

def getEnvs(pairs):
    envs = {
        'crop_0': 'mixed',
        'crop_1': 'tall',
        'crop_2': 'wide',
        'crop_3': 'residential',
    }
    
    image_envs = {}
    for tif_path, _ in pairs:
        for env, type in envs.items():
            if env in tif_path.stem:
                image_envs[tif_path.stem] = type

    return image_envs

def execute() -> None:
    buildings = loadTrainingBuildings(BUILDINGS_GEOJSON)
    pairs = pairImagesWithFootprints(buildings, DATA_DIR)
    image_gsds = getGSDs(pairs)
    image_env = getEnvs(pairs)
    
    tif_path, footprints = random.choice(pairs)
    print(f"random image: {tif_path.name}  ({len(footprints)} labeled buildings)")
    
    available = [spec for spec in MODELS if spec.path.exists()]
    for spec in MODELS:
        if not spec.path.exists():
            print(f"skip missing: {spec.name}")
    if not available:
        raise SystemExit("no model files found")

    device = getDevice("auto")
    print(f"device={device}  score>={SCORE_THRESH}  iou>={IOU_THRESH}")

    with rasterio.open(tif_path) as src:
        truth = groundTruthPolygons(src, footprints)
        print(f"imagery={src.width}x{src.height} crs={src.crs} polygons={len(truth)}")
        frames: dict[str, gpd.GeoDataFrame] = {}
        metrics: dict[str, tuple] = {}
        print(f"{'model':<24} {'n_pred':>7} {'recall':>8} {'precision':>10}")
        for spec in available:
            gdf = runModel(spec, src, device)
            predicted = [] if gdf.empty else [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
            recall, precision, tp, fp = precisionRecall(predicted, truth)
            frames[spec.name] = gdf
            metrics[spec.name] = (recall, precision, tp, fp)
            print(f"{spec.name:<24} {len(predicted):>7} {recall:>8.3f} {precision:>10.3f}")

        drawOverview(src, truth, frames, metrics, OUT_DIR / f"{tif_path.stem}_comparison.png")
    
    results = []
    for tif_path, footprints in tqdm(pairs, desc='test on all images', ncols=100):
        with rasterio.open(tif_path) as src:
            truth = groundTruthPolygons(src, footprints)
            for spec in available:
                gdf = runModel(spec, src, device, False)
                predicted = [] if gdf.empty else [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
                recall, precision, _, _ = precisionRecall(predicted, truth)
                iou, dice = iouDice(predicted, truth)
                results.append({
                    "image_name": tif_path.stem,
                    "model": spec.name,
                    "recall": recall,
                    "precision": precision,
                    "IoU": iou,
                    "dice": dice,
                })

    results = pd.DataFrame(results, columns=["image_name", "model", "recall", "precision", "IoU", "dice"])
    results["gsd"] = results["image_name"].map(image_gsds)
    results["env"] = results["image_name"].map(image_env)
    print("\nPer-image results:\n")
    print(results.to_string(index=False))
    
    mean_results = results.groupby("model")[["recall", "precision", "IoU", "dice"]].mean().reset_index()
    print("\nMean metrics per model:\n")
    print(mean_results.to_string(index=False))

    mean_results = results.groupby("image_name")[["recall", "precision", "IoU", "dice"]].mean().reset_index()
    print("\nMean metrics per image:\n")
    print(mean_results.to_string(index=False))
    
    mean_results = results.groupby("env")[["recall", "precision", "IoU", "dice"]].mean().reset_index()
    print("\nMean metrics per env:\n")
    print(mean_results.to_string(index=False))

    mean_results = results.groupby(["env", "model"])[["recall", "precision", "IoU", "dice"]].mean().reset_index()
    print("\nMean metrics per env per model:\n")
    print(mean_results.to_string(index=False))

    mean_results = results.groupby("gsd")[["recall", "precision", "IoU", "dice"]].mean().reset_index()
    print("\nMean metrics per gsd:\n")
    print(mean_results.to_string(index=False))

if __name__ == "__main__":
    execute()