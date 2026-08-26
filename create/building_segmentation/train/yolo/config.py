import torch
import argparse
from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    name: str
    device: str
    epochs: int
    batch_size: int
    imgsz: int
    model: str
    workers: int
    patience: int

MAC = RunConfig(
    name="mac",
    device="mps",
    epochs=1,
    batch_size=2,
    imgsz=512,
    model="yolo11n-seg.pt",
    workers=0,
    patience=1,
)

PC = RunConfig(
    name="pc",
    device="cuda",
    epochs=80,
    batch_size=8,
    imgsz=512,
    model="yolo11m-seg.pt",
    workers=4,
    patience=20,
)

CONFIGS = {"mac": MAC, "pc": PC}

def defaultConfigName() -> str:
    return "pc" if torch.cuda.is_available() else "mac"

def getConfig(name: str | None = None) -> RunConfig:
    key = name or defaultConfigName()
    if key not in CONFIGS:
        raise SystemExit(f"unknown config '{key}'. choose one of: {', '.join(CONFIGS)}")
    
    return CONFIGS[key]

def parseConfig() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train YOLO building-footprint segmentation")
    parser.add_argument("--config", choices=tuple(CONFIGS), default=None, help="mac: M4 smoke test. pc: CUDA training. default: pc if CUDA is available, else mac.")
    
    return getConfig(parser.parse_args().config)