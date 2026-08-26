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
        raise SystemExit(f"{DATA_YAML.name} is missing - run generate_data.py first.")

    device = ultralyticsDevice(config.device)
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
    print(f' map: {round(metrics.seg.map, 4)}, map50: {round(metrics.seg.map50, 4)}')
    print(f' weights: {WEIGHTS_PATH.name}')

if __name__ == "__main__":
    execute(parseConfig())