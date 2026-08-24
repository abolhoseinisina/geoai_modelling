import cv2
import time
import torch
import rasterio
import numpy as np
import geopandas as gpd
from pathlib import Path
from PIL import Image, ImageDraw
from shapely.strtree import STRtree
from rasterio.windows import Window
from shapely.geometry import Polygon
from rasterio.enums import Resampling
from rasterio.transform import Affine
from config import RunConfig, parseConfig
from common import DATA_DIR, FINETUNED_PATH, HERE, PRETRAINED_PATH, TILE_SIZE, buildModel, getRasterResolution, getDevice, getSamplingScales, getWindowOrigins

TIF_PATH = DATA_DIR / "Winnipeg_SU_2025_crop_3.tif"
OUT_DIR = HERE / "compare_winnipeg"
OVERLAP = 64
SCORE_THRESH = 0.5
MASK_THRESH = 0.5
NMS_IOU = 0.4
MIN_AREA_M2 = 10.0  # drop tiny fragments after polygonisation
MAX_AREA_M2 = 1500.0  # drop absurd blobs (whole fields) that are clearly not buildings
OVERVIEW_SIZE = 1800  # longest side of the comparison PNG
MODELS = [("pretrained", PRETRAINED_PATH), ("finetuned", FINETUNED_PATH)]

def convertMask2Polygonpx(mask: np.ndarray) -> list[tuple[float, float]] | None:
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

def getPolygonIoU(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0

def performNMS(polygons: list[Polygon], scores: list[float], iou_thresh: float) -> list[int]:
    order = sorted(range(len(polygons)), key=lambda i: scores[i], reverse=True)
    tree = STRtree(polygons)
    kept: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue

        kept.append(i)
        for j in tree.query(polygons[i]):
            j = int(j)
            if j == i or j in suppressed:
                continue

            if scores[j] > scores[i]:
                continue

            if getPolygonIoU(polygons[i], polygons[j]) >= iou_thresh:
                suppressed.add(j)

    return kept

@torch.no_grad()
def applyModel2Tile(model, image_chw: torch.Tensor, device) -> list[tuple[Polygon, float]]:
    out = model([image_chw.to(device)])[0]
    keep = out["scores"] >= SCORE_THRESH
    if not keep.any():
        return []

    boxes = out["boxes"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    masks = out["masks"][keep, 0].cpu().numpy() > MASK_THRESH
    results = []
    for mask, score, box in zip(masks, scores, boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cropped = np.zeros_like(mask, dtype=np.uint8)
        y1, y2 = max(y1, 0), min(y2, mask.shape[0])
        x1, x2 = max(x1, 0), min(x2, mask.shape[1])
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        ring = convertMask2Polygonpx(cropped)
        if ring is None:
            continue

        results.append((Polygon(ring), float(score)))
    
    return results

def convertPixelToPolygon(poly: Polygon, transform) -> Polygon:
    return Polygon([transform * (px, py) for px, py in poly.exterior.coords])

def applyModel2Raster(model, src, device) -> gpd.GeoDataFrame:
    scale_x, scale_y, target_gsd = getSamplingScales(src)
    source_width = TILE_SIZE * scale_x
    source_height = TILE_SIZE * scale_y
    xs = getWindowOrigins(src.width, source_width, (TILE_SIZE - OVERLAP) * scale_x)
    ys = getWindowOrigins(src.height, source_height, (TILE_SIZE - OVERLAP) * scale_y)
    total = len(xs) * len(ys)
    polygons: list[Polygon] = []
    scores: list[float] = []
    areas_m2: list[float] = []
    done = 0
    started = time.time()

    for y0 in ys:
        for x0 in xs:
            window = Window(x0, y0, source_width, source_height)
            transform = src.window_transform(window) * Affine.scale(scale_x, scale_y)
            array = src.read([1, 2, 3], window=window, out_shape=(3, TILE_SIZE, TILE_SIZE), resampling=Resampling.bilinear, boundless=True, fill_value=0)
            if (array.max(axis=0) == 0).mean() > 0.6:
                done += 1
                continue

            tensor = torch.from_numpy(array).float().div_(255)
            for poly_px, score in applyModel2Tile(model, tensor, device):
                area_m2 = poly_px.area * target_gsd**2
                if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
                    continue

                poly_map = convertPixelToPolygon(poly_px, transform)
                if not poly_map.is_valid:
                    continue
                
                polygons.append(poly_map)
                scores.append(score)
                areas_m2.append(area_m2)

            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - started
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(f"  tile {done}/{total}  dets={len(polygons)}  {rate:.1f} tiles/s  ETA {eta:.0f}s", flush=True)

    if not polygons:
        return gpd.GeoDataFrame({"score": [], "area_m2": []}, geometry=[], crs=src.crs)

    keep = performNMS(polygons, scores, NMS_IOU)
    geoms = [polygons[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    kept_areas = [areas_m2[i] for i in keep]
    return gpd.GeoDataFrame({"score": kept_scores, "area_m2": kept_areas}, geometry=geoms, crs=src.crs)

def drawOverview(src, frames: dict[str, gpd.GeoDataFrame], out_path: Path) -> None:
    scale = max(src.width, src.height) / OVERVIEW_SIZE
    out_w = max(1, int(round(src.width / scale)))
    out_h = max(1, int(round(src.height / scale)))
    rgb = src.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
    base = np.moveaxis(rgb, 0, -1)
    overview_transform = src.transform * rasterio.Affine.scale(scale)
    colors = {"pretrained": (255, 60, 60), "finetuned": (40, 200, 80)}
    panels = []
    for name, gdf in frames.items():
        panel = Image.fromarray(base.copy())
        draw = ImageDraw.Draw(panel, "RGBA")
        color = colors[name]
        fill = (*color, 70)
        outline = (*color, 220)
        if not gdf.empty:
            inv = ~overview_transform
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue

                parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                for part in parts:
                    if part.geom_type != "Polygon":
                        continue
                    pixels = [
                        (inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f)
                        for x, y in part.exterior.coords
                    ]
                    draw.polygon(pixels, fill=fill, outline=outline)
        draw.rectangle([0, 0, out_w - 1, 28], fill=(0, 0, 0, 160))
        draw.text((8, 6), f"{name}  ({len(gdf)} buildings)", fill=(255, 255, 255, 255))
        panels.append(panel)

    canvas = Image.new("RGB", (out_w * 2 + 8, out_h), (20, 20, 20))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (out_w + 8, 0))
    canvas.save(out_path, quality=92)
    print(f"wrote {out_path}")


def execute(config: RunConfig) -> None:
    if not TIF_PATH.exists():
        raise SystemExit(f"missing imagery: {TIF_PATH}")
    
    for _, path in MODELS:
        if not path.exists():
            raise SystemExit(f"missing weights: {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = getDevice(config.device)
    print(f"config={config.name}  device={device}  tile={TILE_SIZE}  score>={SCORE_THRESH}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"imagery={TIF_PATH}")
    frames: dict[str, gpd.GeoDataFrame] = {}
    with rasterio.open(TIF_PATH) as src:
        print(f"size={src.width}x{src.height}  crs={src.crs}  res={src.res}")
        x_gsd, y_gsd = getRasterResolution(src)
        scale_x, scale_y, target_gsd = getSamplingScales(src)
        print(f"native GSD={x_gsd * 100:.1f}x{y_gsd * 100:.1f} cm/px  model GSD={target_gsd * 100:.1f} cm/px  source scale={scale_x:.2f}x{scale_y:.2f}")
        for name, weights in MODELS:
            print(f"\n=== {name}: {weights.name} ===")
            model = buildModel(weights_path=weights)
            model.to(device).eval()
            gdf = applyModel2Raster(model, src, device)
            out_geojson = OUT_DIR / f"{name}_footprints.geojson"
            gdf.to_file(out_geojson, driver="GeoJSON")
            print(f"  {len(gdf)} buildings after NMS  median area {gdf['area_m2'].median():.1f} m²  -> {out_geojson.name}")
            frames[name] = gdf
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        drawOverview(src, frames, OUT_DIR / "comparison_overview.png")

    pretrained_n = len(frames["pretrained"])
    finetuned_n = len(frames["finetuned"])
    print(f"\nsummary: pretrained={pretrained_n}  finetuned={finetuned_n}  delta={finetuned_n - pretrained_n:+d}")
    print(f"outputs in {OUT_DIR}")

if __name__ == "__main__":
    execute(parseConfig())