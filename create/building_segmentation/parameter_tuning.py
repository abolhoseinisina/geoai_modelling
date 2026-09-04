import time
import pandas as pd
from pathlib import Path
from device import getDevice
from tiling import generateTiles, generateDataYML
from config import parseYOLOConfig, parseFinetuneConfig
from train.yolo.common import trainYOLOModel, validateYOLOModel
from finetune.common import finetuneMaskRCNN, validateMaskRCNNModel

SEED = 42

TILE_SIZES = [320, 512, 640, 960]
OVERLAPS = [0, 32, 64, 128, 192]
MODEL_TYPES = ['YOLO', 'MASK-RCNN']

REPO = Path(__file__).resolve().parent.parent.parent
TRAINING_IMAGES_DIR = REPO / 'datasets/training_datasets/building_segmentation_202608'
TRAINING_DETECTION_FILE = TRAINING_IMAGES_DIR / 'training_buildings.geojson'
TRAINING_TILES_DIR = REPO / 'create/building_segmentation/tiles/training'
TRAINING_TILE_INDEX_FILE = TRAINING_TILES_DIR / "index.json"
TRAINING_DATASET_DIR = TRAINING_TILES_DIR / 'dataset'
TRAINING_DATASET_FILE = TRAINING_DATASET_DIR / 'data.yaml'

VALIDATING_IMAGES_DIR = REPO / 'datasets/validating_datasets/building_segmentation_202608'
VALIDATING_DETECTION_FILE = VALIDATING_IMAGES_DIR / 'validation_buildings.geojson'
VALIDATING_TILES_DIR = REPO / 'create/building_segmentation/tiles/validating'
VALIDATING_TILE_INDEX_FILE = VALIDATING_TILES_DIR / "index.json"
VALIDATING_DATASET_DIR = VALIDATING_TILES_DIR / 'dataset'
VALIDATING_DATASET_FILE = VALIDATING_DATASET_DIR / 'data.yaml'

OUTPUT_DIR = REPO / 'create/building_segmentation/output'
OUTPUT_MODELS_DIR = OUTPUT_DIR / 'models'

def drawComparisonChart(results, file_path):
    return

def trainYolo(data_yaml: Path, tile_size: int, output_model_dir: Path):
    config = parseYOLOConfig()
    device = getDevice(config.device)
    model = trainYOLOModel(config.model, data_yaml, device, config.epochs, tile_size, config.batch_size, config.workers, config.patience, output_model_dir)
    return model

def validateYolo(model, validation_tiles_dir, validation_tile_index, tile_size):
    return validateYOLOModel(model, validation_tiles_dir, validation_tile_index, tile_size, 0.5)

def trainMaskRCNN(tile_index: list[dict], tiles_dir: Path, tile_size: int, overlap: int, output_models_dir: Path):
    config = parseFinetuneConfig()
    device = getDevice(config.device)
    model = finetuneMaskRCNN(config.pretrained_model_path, tile_index, tiles_dir, SEED, 0.2, config.batch_size, config.num_workers, config.pin_memory, device, config.epochs, config.learning_rate, output_models_dir)
    return model

def validateMaskRCNN(model, validation_tiles_dir, validation_tile_index, tile_size, overlap):
    config = parseFinetuneConfig()
    device = getDevice(config.device)
    results = validateMaskRCNNModel(model, validation_tiles_dir, validation_tile_index, device, 0.5)
    return results

def train(model_type, tile_index, tiles_dir, data_file, tile_size, overlap, output_models_dir):
    if model_type == 'YOLO':
        return trainYolo(data_file, tile_size, output_models_dir)

    elif model_type == 'MASK-RCNN':
        return trainMaskRCNN(tile_index, tiles_dir, tile_size, overlap, output_models_dir)

    raise SystemExit('Error: Wrong "model_type" value.')

def validate(model_type, model, validation_tiles_dir, validation_tile_index, tile_size, overlap, gsd_m):
    if model_type == 'YOLO':
        return validateYolo(model, validation_tiles_dir, validation_tile_index, tile_size)

    elif model_type == 'MASK-RCNN':
        return validateMaskRCNN(model, validation_tiles_dir, validation_tile_index, tile_size, overlap)

    raise SystemExit('Error: Wrong "model_type" value.')

def main():
    gsd_m = 0.1
    results = []
    
    for tile_size in TILE_SIZES:
        for overlap in OVERLAPS:
            training_tile_index = generateTiles(TRAINING_IMAGES_DIR, TRAINING_DETECTION_FILE, TRAINING_TILES_DIR, TRAINING_TILE_INDEX_FILE, tile_size, overlap, gsd_m=gsd_m)
            training_data_file = generateDataYML(training_tile_index, TRAINING_TILES_DIR, TRAINING_DATASET_DIR, TRAINING_DATASET_FILE, 0.2, SEED)
            validation_tile_index = generateTiles(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE, tile_size, overlap, gsd_m=gsd_m)
            for model_type in MODEL_TYPES:
                model = train(model_type, training_tile_index, TRAINING_TILES_DIR, training_data_file, tile_size, overlap, OUTPUT_MODELS_DIR)

                t0 = time.time()
                model_performace = validate(model_type, model, VALIDATING_TILES_DIR, validation_tile_index, tile_size, overlap, gsd_m)
                duration = time.time() - t0
                
                model_performace = model_performace[model_performace['actual'] > 0]
                sum_columns = ['actual', 'predicted', 'true_positives', 'false_positives']
                mean_columns = ['precision', 'recall', 'iou', 'dice']
                group = model_performace.groupby('source')
                sums = group[sum_columns].sum()
                means = group[mean_columns].mean()
                stats = pd.concat([sums, means], axis=1)
                for row, source in stats.iterrows():
                    results.append({
                        'tile_size': tile_size, 
                        'overlap': overlap, 
                        'model': model_type, 
                        'validation_duration_sec': round(duration, 2),
                        'source': row, 
                        'actual': int(source['actual']), 
                        'predicted': int(source['predicted']), 
                        'precision': round(source['precision'], 4), 
                        'recall': round(source['recall'], 4), 
                        'iou': round(source['iou'], 4), 
                        'dice': round(source['dice'], 4)
                    })
               
                results_df = pd.DataFrame(results)
                results_df.to_csv('output/parameter_tuning.csv')
    
    drawComparisonChart(results_df, 'output/parameter_tuning.jpg')

if __name__ == '__main__':
    main()