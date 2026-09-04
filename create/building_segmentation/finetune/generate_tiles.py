import sys
from pathlib import Path

_BUILDING_SEG = Path(__file__).resolve().parent.parent
if str(_BUILDING_SEG) not in sys.path:
    sys.path.insert(0, str(_BUILDING_SEG))

from tiling import executeTiling
from common import HERE, TILE_INDEX, TILES_DIR

def execute() -> None:
    executeTiling(TILES_DIR, TILE_INDEX, HERE / "output/alignment_check.png")

if __name__ == "__main__":
    execute()