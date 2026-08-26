import json
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from common import DATA_YAML, DATASET_DIR, TILE_INDEX, TILES_DIR

SEED = 42
VAL_FRACTION = 0.15

def yoloLabelLines(record: dict) -> list[str]:
    size = record["size"]
    lines = []
    for rings in record["polygons"]:
        exterior = rings[0]
        if len(exterior) < 3:
            continue
        
        coords = np.clip(np.array(exterior, dtype=np.float64) / size, 0, 1)
        if len(coords) > 3 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        
        if len(coords) < 3:
            continue
        
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
        lines.append(f"0 {coord_str}")
    
    return lines

def writeAlignmentCheck(record: dict, image_path: Path, label_path: Path, output_path: Path) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas = ImageDraw.Draw(overlay, "RGBA")
    h, w = image.shape[:2]
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = list(map(float, line.split()[1:]))
            if len(parts) < 6:
                continue

            pts = [(x * w, y * h) for x, y in zip(parts[0::2], parts[1::2])]
            canvas.polygon(pts, fill=(255, 40, 40, 102), outline=(0, 255, 0, 255))

    out = Image.alpha_composite(base, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)
    print(f'Check "{output_path.name}" ({record["image"]}, {len(record["polygons"])} buildings): the green outlines should sit on roof edges.')

def execute() -> None:
    if not TILE_INDEX.exists():
        raise SystemExit(f"{TILE_INDEX} is missing - run generate_tiles.py first.")

    records = json.loads(TILE_INDEX.read_text())
    rng = random.Random(SEED)
    shuffled = list(records)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * VAL_FRACTION))
    splits = {"val": shuffled[:cut], "train": shuffled[cut:]}

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split, split_records in splits.items():
        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for record in split_records:
            src = TILES_DIR / record["image"]
            dest_image = image_dir / Path(record["image"]).name
            shutil.copy2(src, dest_image)
            (label_dir / f"{dest_image.stem}.txt").write_text("\n".join(yoloLabelLines(record)))

    DATA_YAML.write_text(
        "\n".join(
            [
                f"path: {DATASET_DIR.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: building",
                "",
            ]
        )
    )
    print(f"Wrote {DATA_YAML} ({len(splits['train'])} train, {len(splits['val'])} val)")

    labeled = [record for record in splits["train"] if record["polygons"]]
    sample = rng.choice(labeled)
    stem = Path(sample["image"]).stem
    writeAlignmentCheck(
        sample,
        DATASET_DIR / "images/train" / Path(sample["image"]).name,
        DATASET_DIR / "labels/train" / f"{stem}.txt",
        Path(__file__).resolve().parent / "output/yolo_alignment_check.png",
    )

if __name__ == "__main__":
    execute()