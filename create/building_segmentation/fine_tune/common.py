import torch
from torch import nn
from pyproj import Geod
from pathlib import Path
from rasterio.warp import transform as transform_coordinates
from torchvision.models.detection import maskrcnn_resnet50_fpn

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "training_dataset/building_segmentation_202608"
BUILDINGS_GEOJSON = DATA_DIR / "training_buildings.geojson"
PRETRAINED_PATH = HERE.parent.parent.parent / "models/building_segmentation/building_footprints_usa.pth"
FINETUNED_PATH = HERE / "output/finetuned_building_footprints_usa.pth"
ONNX_PATH = HERE / "output/finetuned_building_footprints_usa.onnx"

TILES_DIR = HERE / "tiles"
TILE_INDEX = TILES_DIR / "index.json"
TILE_SIZE = 512
NUM_CLASSES = 2  # 0 = background, 1 = building

MIN_GSD_M = 0.10
_GEOD = Geod(ellps="WGS84")
_BACKBONE_STAGES = ("layer4", "layer3", "layer2", "layer1", "conv1")

def getRasterResolution(raster) -> tuple[float, float]:
    if raster.crs is None:
        raise ValueError(f"{raster.name} has no CRS; ground resolution is unknown")

    col = (raster.width - 1) / 2
    row = (raster.height - 1) / 2
    points = [raster.transform * (col, row), raster.transform * (col + 1, row), raster.transform * (col, row + 1)]
    longitudes, latitudes = transform_coordinates(raster.crs, "EPSG:4326", [point[0] for point in points], [point[1] for point in points])
    _, _, x_m = _GEOD.inv(longitudes[0], latitudes[0], longitudes[1], latitudes[1])
    _, _, y_m = _GEOD.inv(longitudes[0], latitudes[0], longitudes[2], latitudes[2])
    return abs(float(x_m)), abs(float(y_m))

def getSamplingScales(raster) -> tuple[float, float, float]:
    x_gsd, y_gsd = getRasterResolution(raster)
    target_gsd = max(MIN_GSD_M, x_gsd, y_gsd)
    return target_gsd/x_gsd, target_gsd/y_gsd, target_gsd

def getWindowOrigins(total: int, span: float, step: float) -> list[float]:
    if total <= span:
        return [0.0]

    limit = total - span
    origins: list[float] = []
    origin = 0.0
    while origin <= limit + 1e-6:
        origins.append(origin)
        origin += step
    
    if not origins or abs(origins[-1] - limit) > 1e-6:
        origins.append(limit)
    
    return origins

def getDevice(preference: str = "cpu") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")

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