import sys
import warnings
from pathlib import Path
from ultralytics import YOLO
from config import RunConfig, parseConfig
from common import DATA_YAML, OUTPUT_DIR, WEIGHTS_PATH

_BUILDING_SEG = Path(__file__).resolve().parents[2]
if str(_BUILDING_SEG) not in sys.path:
    sys.path.insert(0, str(_BUILDING_SEG))

from device import getDevice

warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

def ultralyticsDevice(preference: str):
    device = getDevice(preference)
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    
    return device.type

def execute(config: RunConfig) -> None:
    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} is missing - run generate_data.py first.")

    device = ultralyticsDevice(config.device)
    print(f"Train YOLO ({config.name}): model={config.model}  device={device}  epochs={config.epochs}  batch={config.batch_size}  imgsz={config.imgsz}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.model)
    model.train(
        data=str(DATA_YAML),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch_size,
        device=device,
        workers=config.workers,
        patience=config.patience,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        plots=True,
    )

    best = YOLO(str(WEIGHTS_PATH))
    metrics = best.val(data=str(DATA_YAML), imgsz=config.imgsz, device=device, workers=config.workers)
    print("map", metrics.seg.map, "map50", metrics.seg.map50)
    print(f"weights: {WEIGHTS_PATH}")

if __name__ == "__main__":
    execute(parseConfig())