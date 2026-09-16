import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from shapely.geometry import Polygon

from nms import georeferencePolygons, performNMS
from accuracy import getIoUDice, getPrecisionRecall

def trainYOLOModel(yolo_base_model, data_yaml: Path, device, epochs: int, tile_size: int, batch_size: int, workers: int, patience: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(yolo_base_model)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=tile_size, batch=batch_size, device=device, workers=workers, patience=patience, project=str(output_dir), name="train", exist_ok=True, plots=True)
    return YOLO(str(output_dir / "train/weights/best.pt"))

def validateYOLOModel(model: YOLO, validation_tiles_dir: Path, validation_tile_index: dict, tile_size: int, score_threshold: float, truth_by_source: dict[str, list[Polygon]], nms_iou_thresh: float, accuracy_iou_thresh: float):
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