import torch
import argparse
from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    name: str
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    pin_memory: bool

MAC = RunConfig(
    name="mac",
    device="cpu",
    epochs=1,
    batch_size=2,
    learning_rate=0.002,
    num_workers=0,
    pin_memory=False,
)

PC = RunConfig(
    name="pc",
    device="cuda",
    epochs=20,
    batch_size=8,
    learning_rate=0.01,
    num_workers=4,
    pin_memory=True,
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
    parser = argparse.ArgumentParser(description="Fine-tune building footprints")
    parser.add_argument(
        "--config",
        choices=tuple(CONFIGS),
        default=None,
        help="mac: laptop smoke test. pc: CUDA training (RTX 4080 Ti). default: pc if CUDA is available, else mac.",
    )
    return getConfig(parser.parse_args().config)