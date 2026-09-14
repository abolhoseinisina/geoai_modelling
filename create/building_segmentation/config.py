import json
import torch
from pyproj import Geod
from pathlib import Path
from train.yolo.config import YOLOConfig
from finetune.config import FineTuneConfig

SEED = 42

IMAGE_NAME_COLUMN = 'image_name'
NEGATIVE_RATIO = 0.5
MAX_BLANK_FRACTION = 0.5
MIN_INSTANCE_AREA_PX = 24

GEOD = Geod(ellps="WGS84")

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

def defaultConfigName() -> str:
    return "pc" if torch.cuda.is_available() else "mac"

def getFinalModelConfig():
    config_path = 'final_model_config.json'
    with open(config_path, "r") as f:
        config = json.load(f)

    key = defaultConfigName()
    config['YOLO'] = YOLOConfig(key)
    config['MASK-RCNN'] = FineTuneConfig(key)
    
    return config