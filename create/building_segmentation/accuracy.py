from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.validation import make_valid

def _safePolygon(poly: Polygon) -> Polygon | None:
    if poly.is_empty or len(poly.exterior.coords) < 4:
        return None

    fixed = make_valid(poly)
    if fixed.geom_type == "Polygon" and not fixed.is_empty:
        return fixed
    
    if fixed.geom_type == "MultiPolygon":
        return max(fixed.geoms, key=lambda g: g.area)
    
    return None

def _safePolygons(polys: list[Polygon]) -> list[Polygon]:
    out = []
    for poly in polys:
        fixed = _safePolygon(poly)
        if fixed is not None:
            out.append(fixed)
    
    return out

def getPolygonIoU(a: Polygon, b: Polygon) -> float:
    a, b = _safePolygon(a), _safePolygon(b)
    if a is None or b is None:
        return 0.0

    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0

def getPrecisionRecall(predicted: list[Polygon], truth: list[Polygon], iou_thresh: float) -> tuple[float, float, int, int]:
    used = set()
    true_positives = 0
    for truth_poly in truth:
        best_i, best_iou = -1, 0.0
        for i, pred_poly in enumerate(predicted):
            if i in used:
                continue
            
            iou = getPolygonIoU(truth_poly, pred_poly)
            if iou > best_iou:
                best_iou, best_i = iou, i
        
        if best_iou >= iou_thresh:
            true_positives += 1
            used.add(best_i)
    
    false_positives = max(len(predicted) - true_positives, 0)
    recall = true_positives / max(len(truth), 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    return recall, precision, true_positives, false_positives

def getIoUDice(predicted: list[Polygon], truth: list[Polygon]) -> tuple[float, float]:
    predicted = _safePolygons(predicted)
    truth = _safePolygons(truth)

    if not predicted and not truth:
        return 1.0, 1.0
    
    if not predicted or not truth:
        return 0.0, 0.0

    pred_union = unary_union(predicted)
    truth_union = unary_union(truth)
    if pred_union.is_empty or truth_union.is_empty:
        return 0.0, 0.0

    intersection = pred_union.intersection(truth_union).area
    union = pred_union.union(truth_union).area
    iou = intersection / union if union > 0 else 0.0
    denom = pred_union.area + truth_union.area
    dice = (2.0 * intersection / denom) if denom > 0 else 0.0
    return iou, dice