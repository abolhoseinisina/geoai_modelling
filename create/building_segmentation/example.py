import pandas as pd
from ultralytics import YOLO

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon
from itertools import islice
import pyproj
from shapely.ops import transform

from device import getDevice
from tiling import generateTiles, loadGroundTruthBySource
from train.yolo.utils import apply2Tiles as applyYOLO2Tiles
from finetune.utils import buildModel, apply2Tiles as applyRCNN2Tiles
from compare import MODELS, ModelSpec, applyMaskRCNNOnnx2Tiles, applyXunetOnnx2Tiles, ortSession
from config import getFinalModelConfig, SEED, VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE

EXAMPLE_IMAGE_NAME = 'Winnipeg_SU_2025_crop_3.tif'

def drawResults(validation_tiles_dir, predictions, ground_truth):
    image_path = validation_tiles_dir / EXAMPLE_IMAGE_NAME
    with rasterio.open(image_path) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / np.ptp(img)
        raster_crs = src.crs
        transform_raster = src.transform

    if raster_crs is not None and raster_crs.to_epsg() != 4326:
        proj_to_raster = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True).transform
    else:
        proj_to_raster = None

    def lonlat_poly_to_pixels(poly):
        poly_proj = transform(proj_to_raster, poly) if proj_to_raster else poly
        coords = np.array(poly_proj.exterior.coords)
        rows, cols = rasterio.transform.rowcol(transform_raster, coords[:, 0], coords[:, 1])
        return np.array(cols), np.array(rows)

    model_names = list(predictions.keys())
    nrows, ncols = 2, 3
    total_plots = nrows * ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 10))
    axs = axs.flatten()
    shown_models = list(islice(model_names, total_plots))

    # For making legend handles, store first line references
    ground_truth_handle = None
    prediction_handle = None

    for idx in range(total_plots):
        if idx >= len(shown_models):
            continue
        
        ax = axs[idx]
        ax.imshow(img)
        ax.set_axis_off()
        model = shown_models[idx]
        ax.set_title(model, fontsize=14)
        
        for poly in ground_truth.get(EXAMPLE_IMAGE_NAME, []):
            x, y = lonlat_poly_to_pixels(poly)
            line, = ax.plot(x, y, color='lime', linewidth=1.5, alpha=0.5, label='Ground Truth' if idx==0 else "")
            ground_truth_handle = line
            
        for poly in predictions.get(model, []):
            x, y = lonlat_poly_to_pixels(poly)
            line, = ax.plot(x, y, color='red', linewidth=2, alpha=0.7, label='Prediction' if idx==0 else "")
            prediction_handle = line
    
    fig.legend([ground_truth_handle, prediction_handle], ['Ground Truth', 'Prediction'], loc='center right', fontsize=14)
    plt.suptitle(f"Example: {EXAMPLE_IMAGE_NAME.split('.')[0]}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.savefig(f'output/models/example_{EXAMPLE_IMAGE_NAME.split('.')[0]}.jpg', dpi=500)
    plt.close()

def apply(model_spec: ModelSpec, validation_tile_dir: str, validation_tile_index: dict, device, nms_iou_threshold, accuracy_iou_threshold):
    score_threshold = 0.5
    tiles = pd.DataFrame(validation_tile_index)
    if model_spec.model_type == "MASK-RCNN":
        model = buildModel(weights_path=model_spec.path, num_classes=2, image_size=model_spec.tile_size)
        model.to(device).eval()
        return applyRCNN2Tiles(model, validation_tile_dir, tiles, device, score_threshold, nms_iou_threshold)
    
    elif model_spec.model_type == "MASK-RCNN-ONNX":
        mask_threshold = 0.5
        session = ortSession(model_spec.path)
        input_name = session.get_inputs()[0].name
        return applyMaskRCNNOnnx2Tiles(session, validation_tile_dir, tiles, score_threshold, nms_iou_threshold, input_name, mask_threshold)
    
    elif model_spec.model_type == "XUNET-ONNX":
        seg_thresh = 0.5
        min_segment_px = 11
        session = ortSession(model_spec.path)
        input_name = session.get_inputs()[0].name
        return applyXunetOnnx2Tiles(session, validation_tile_dir, tiles, nms_iou_threshold, input_name, seg_thresh, min_segment_px, peak=0)
    
    elif model_spec.model_type == "YOLO-ONNX":
        model = YOLO(model_spec.path, task='segment')
        return applyYOLO2Tiles(model, validation_tile_dir, tiles, model_spec.tile_size, score_threshold, nms_iou_threshold) 
    
    elif model_spec.model_type == "YOLO":
        model = YOLO(model_spec.path, task='segment')
        return applyYOLO2Tiles(model, validation_tile_dir, tiles, model_spec.tile_size, score_threshold, nms_iou_threshold) 
    
    else:
        raise SystemExit(f"unknown model kind: {model_spec.kind}")

def main():
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
    ground_truth = {EXAMPLE_IMAGE_NAME: truth_by_source.get(EXAMPLE_IMAGE_NAME, [])}
    
    results = {}
    for model_spec in available_models:
        print(f' + Model: {model_spec.name}')
        validation_tile_index = generateTiles(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE, model_spec.tile_size, model_spec.overlap, gsd_m=model_spec.gsd_m, seed=SEED)
        example_tile_index = [tile for tile in validation_tile_index if tile['source'] == EXAMPLE_IMAGE_NAME]
        predicted = apply(model_spec, VALIDATING_TILES_DIR, example_tile_index, device, config['nms_iou_threshold'], config['accuracy_iou_threshold'])
        results[model_spec.name] = predicted
        
    drawResults(VALIDATING_IMAGES_DIR, results, ground_truth)

if __name__ == '__main__':
    main()

