import torch
import argparse
from dataclasses import dataclass

@dataclass(frozen=True)
class YOLORunConfig:
    name: str
    device: str
    epochs: int
    batch_size: int
    imgsz: int
    model: str
    workers: int
    patience: int

@dataclass(frozen=True)
class FTRunConfig:
    name: str
    pretrained_model_path: str
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    pin_memory: bool

YOLO_MAC = YOLORunConfig(
    name="mac",
    device="mps",
    epochs=1,
    batch_size=2,
    imgsz=512,
    model="yolo26n-seg.pt",
    workers=0,
    patience=1,
)

YOLO_PC = YOLORunConfig(
    name="pc",
    device="cuda",
    epochs=80,
    batch_size=8,
    imgsz=512,
    model="yolo26l-seg.pt",
    workers=4,
    patience=20,
)

FT_MAC = FTRunConfig(
    name="mac",
    pretrained_model_path="/Users/sina/Desktop/FastMap/geoai_modelling/geoai_modelling/models/building_footprints_usa.pth",
    device="cpu",
    epochs=1,
    batch_size=2,
    learning_rate=0.002,
    num_workers=0,
    pin_memory=False,
)

FT_PC = FTRunConfig(
    name="pc",
    pretrained_model_path="/Users/sina/Desktop/FastMap/geoai_modelling/geoai_modelling/models/building_footprints_usa.pth",
    device="cuda",
    epochs=24,
    batch_size=4,
    learning_rate=0.005,
    num_workers=4,
    pin_memory=True,
)

YOLO_CONFIGS = {"mac": YOLO_MAC, "pc": YOLO_PC}
FT_CONFIGS = {"mac": FT_MAC, "pc": FT_PC}

def defaultConfigName() -> str:
    return "pc" if torch.cuda.is_available() else "mac"

def getYOLOConfig() -> YOLORunConfig:
    key = defaultConfigName()
    if key not in YOLO_CONFIGS:
        raise SystemExit(f"unknown config '{key}'. choose one of: {', '.join(YOLO_CONFIGS)}")
    
    return YOLO_CONFIGS[key]

def getFinetuneConfig() -> FTRunConfig:
    key = defaultConfigName()
    if key not in FT_CONFIGS:
        raise SystemExit(f"unknown config '{key}'. choose one of: {', '.join(FT_CONFIGS)}")
    
    return FT_CONFIGS[key]

def parseYOLOConfig() -> YOLORunConfig:
    return getYOLOConfig()

def parseFinetuneConfig() -> FTRunConfig:
    return getFinetuneConfig()