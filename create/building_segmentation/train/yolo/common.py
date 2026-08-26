from pathlib import Path

HERE = Path(__file__).resolve().parent
TILES_DIR = HERE / "tiles"
TILE_INDEX = TILES_DIR / "index.json"
DATASET_DIR = HERE / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
OUTPUT_DIR = HERE / "output"
WEIGHTS_PATH = OUTPUT_DIR / "train/weights/best.pt"
ONNX_PATH = OUTPUT_DIR / "train/weights/best.onnx"
