import sys
import torch
from torch import nn
from pathlib import Path
from torchvision.models.detection import maskrcnn_resnet50_fpn

HERE = Path(__file__).resolve().parent
_BUILDING_SEG = HERE.parent
if str(_BUILDING_SEG) not in sys.path:
    sys.path.insert(0, str(_BUILDING_SEG))

from device import cudaFreeBytes, getDevice, pickCudaDevice  # noqa: E402, F401
from geo import BUILDINGS_GEOJSON, DATA_DIR, TILE_SIZE, getRasterResolution, getSamplingScales, getWindowOrigins  # noqa: E402

PRETRAINED_PATH = HERE.parent.parent.parent / "models/building_footprints_usa.pth"
FINETUNED_PATH = HERE / "output/finetuned_building_footprints_usa.pth"
ONNX_PATH = HERE / "output/finetuned_building_footprints_usa.onnx"
TILES_DIR = HERE / "tiles"
TILE_INDEX = TILES_DIR / "index.json"
NUM_CLASSES = 2  # 0 = background, 1 = building
_BACKBONE_STAGES = ("layer4", "layer3", "layer2", "layer1", "conv1")

def loadAndUnwrapStateDict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def buildModel(weights_path: Path | None = None, num_classes: int = NUM_CLASSES, image_size: int = TILE_SIZE) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes, min_size=image_size, max_size=image_size)
    if weights_path is not None:
        model.load_state_dict(loadAndUnwrapStateDict(weights_path))

    return model

def freezeBackbone(model: nn.Module, trainable_stages: int) -> None:
    keep = _BACKBONE_STAGES[:trainable_stages]
    for name, param in model.backbone.body.named_parameters():
        if not any(name.startswith(stage) for stage in keep):
            param.requires_grad_(False)

def freezeBatchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)