from rasterio.transform import Affine
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid

NMS_IOU = 0.4

def convertPixelToPolygon(poly: Polygon, transform) -> Polygon:
    if not isinstance(transform, Affine):
        transform = Affine(*list(transform)[:6])
    return Polygon([transform * (px, py) for px, py in poly.exterior.coords])

def _polygonIoU(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
    except Exception:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0

def performNMS(polygons: list[Polygon], scores: list[float], iou_thresh: float = NMS_IOU) -> list[int]:
    if not polygons:
        return []

    order = sorted(range(len(polygons)), key=lambda i: scores[i], reverse=True)
    tree = STRtree(polygons)
    kept: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue

        kept.append(i)
        for j in tree.query(polygons[i]):
            j = int(j)
            if j == i or j in suppressed:
                continue
            if scores[j] > scores[i]:
                continue
            if _polygonIoU(polygons[i], polygons[j]) >= iou_thresh:
                suppressed.add(j)

    return kept

def georeferencePolygons(pixel_polys: list[Polygon], scores: list[float], transform) -> tuple[list[Polygon], list[float]]:
    mapped: list[Polygon] = []
    mapped_scores: list[float] = []
    for poly, score in zip(pixel_polys, scores):
        poly_map = convertPixelToPolygon(poly, transform)
        if not poly_map.is_valid:
            poly_map = make_valid(poly_map)
        if poly_map.geom_type == "MultiPolygon":
            poly_map = max(poly_map.geoms, key=lambda g: g.area)
        if poly_map.geom_type != "Polygon" or poly_map.is_empty:
            continue
        mapped.append(poly_map)
        mapped_scores.append(float(score))
    return mapped, mapped_scores
