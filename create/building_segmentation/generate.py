import onnx
import torch
from pathlib import Path
from device import getDevice

from train.yolo.common import trainYOLOModel, validateYOLOModel
from tiling import generateTiles, generateDataYAML, loadGroundTruthBySource
from finetune.common import finetuneMaskRCNN, validateMaskRCNNModel, buildModel
from config import (
    getFinalModelConfig,
    SEED,
    OUTPUT_MODELS_DIR,
    TRAINING_IMAGES_DIR,
    TRAINING_DETECTION_FILE,
    TRAINING_TILES_DIR,
    TRAINING_TILE_INDEX_FILE,
    TRAINING_DATASET_DIR,
    TRAINING_DATASET_FILE,
    VALIDATING_IMAGES_DIR,
    VALIDATING_DETECTION_FILE,
    VALIDATING_TILES_DIR,
    VALIDATING_TILE_INDEX_FILE,
)

def validateYolo(model, validation_tiles_dir, validation_tile_index, tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold):
    return validateYOLOModel(model, validation_tiles_dir, validation_tile_index, tile_size, 0.5, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)

def validateMaskRCNN(model, model_config, validation_tiles_dir, validation_tile_index, tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold):
    device = getDevice(model_config.device)
    return validateMaskRCNNModel(model, validation_tiles_dir, validation_tile_index, device, 0.5, tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)

def validate(model_type, model, model_config, validation_tiles_dir, validation_tile_index, tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold):
    if model_type == 'YOLO':
        return validateYolo(model, validation_tiles_dir, validation_tile_index, tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)

    elif model_type == 'MASK-RCNN':
        return validateMaskRCNN(model, model_config, validation_tiles_dir, validation_tile_index, tile_size, truth_by_source, nms_iou_threshold, accuracy_iou_threshold)

    raise SystemExit('Error: Wrong "model_type" value.')

def trainYolo(model_config, data_yaml: Path, tile_size: int, output_model_dir: Path):
    device = getDevice(model_config.device)
    model = trainYOLOModel(model_config.model, data_yaml, device, model_config.epochs, tile_size, model_config.batch_size, model_config.workers, model_config.patience, output_model_dir)
    return model

def trainMaskRCNN(model_config, tile_index: list[dict], tiles_dir: Path, tile_size: int, output_models_dir: Path):
    device = getDevice(model_config.device)
    train_test_fraction = 0.2
    model = finetuneMaskRCNN(model_config.pretrained_model_path, tile_index, tiles_dir, tile_size, SEED, train_test_fraction, model_config.batch_size, model_config.num_workers, model_config.pin_memory, device, model_config.epochs, model_config.learning_rate, output_models_dir)
    return model

def train(model_type, model_config, tile_index, tiles_dir, data_file, tile_size, output_models_dir):
    if model_type == 'YOLO':
        return trainYolo(model_config, data_file, tile_size, output_models_dir)

    elif model_type == 'MASK-RCNN':
        return trainMaskRCNN(model_config, tile_index, tiles_dir, tile_size, output_models_dir)

    raise SystemExit('Error: Wrong "model_type" value.')

def generateONNX(model_type, model, format: str, tile_size: int, output_path: str):
    if model_type == 'YOLO':
        exported = model.export(format='onnx', imgsz=tile_size, opset=12, simplify=True, dynamic=False)
        print(f"Model save as .onnx in {exported}")

    elif model_type == 'MASK-RCNN':
        buildModel()
        dummy_input = [torch.rand(3, tile_size, tile_size)]
        torch.onnx.export(model, (dummy_input,), str(output_path), dynamo=False, export_params=True, opset_version=17, do_constant_folding=True,
                        input_names=["images"], output_names=["boxes", "labels", "scores", "masks"], 
                        dynamic_axes={"images": {1: "height", 2: "width"}, "boxes": {0: "num_detections"}, "labels": {0: "num_detections"}, "scores": {0: "num_detections"}, "masks": {0: "num_detections"}})

        onnx.checker.check_model(onnx.load(str(output_path)))
        print(f"Model save as .onnx in {output_path.name}")
    
    raise SystemExit('Error: Wrong "model_type" value.')

def main():
    config = getFinalModelConfig()
    training_tile_index = generateTiles(TRAINING_IMAGES_DIR, TRAINING_DETECTION_FILE, TRAINING_TILES_DIR, TRAINING_TILE_INDEX_FILE, config['tile_size'], config['overlap'], gsd_m=config['gsd_minimum'], seed=SEED)
    training_data_file = generateDataYAML(training_tile_index, TRAINING_TILES_DIR, TRAINING_DATASET_DIR, TRAINING_DATASET_FILE, 0.2, SEED)
    validation_tile_index = generateTiles(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE, VALIDATING_TILES_DIR, VALIDATING_TILE_INDEX_FILE, config['tile_size'], config['overlap'], gsd_m=config['gsd_minimum'], seed=SEED)
    truth_by_source = loadGroundTruthBySource(VALIDATING_IMAGES_DIR, VALIDATING_DETECTION_FILE)
    model_type = config['model_type']
    model = train(model_type, config[model_type], training_tile_index, TRAINING_TILES_DIR, training_data_file, config['tile_size'], OUTPUT_MODELS_DIR)
    model_performance = validate(model_type, model, config[model_type], VALIDATING_TILES_DIR, validation_tile_index, config['tile_size'], truth_by_source, config['nms_iou_threshold'], config['accuracy_iou_threshold'])
    generateONNX(model_type, model, 'onnx', config['tile_size'], OUTPUT_MODELS_DIR)

if __name__ == '__main__':
    main()