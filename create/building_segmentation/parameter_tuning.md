# Parameter Tuning Results

Tile size and overlap were swept for YOLO26x-seg and a fine-tuned Mask R-CNN. Metrics are **image-level after polygon NMS**. F1 / precision / recall are instance matches at IoU 0.5. Dice / IoU compare the **union** of predicted footprints to the **union** of labels (coverage of the roof pixels).

**Wide** (`crop_2`) is reported only as a contrast: neither model finds those buildings as instances (F1 ~0.03–0.15) even when Mask R-CNN Dice stays high from over-painting.

Target scenes: **residential**, **mixed**, and **tall** (12 images).

### Environments

| Env | Sample | Images | What the numbers say |
|-----|--------|--------|----------------------|
| **Residential** | <img src="output/parameter_tuning/previews/residential_preview.jpg" alt="residential" width="180" height="120"> | 4 | Both models are usable. Mask R-CNN finds more houses; YOLO is more conservative. Footprints (Dice) favor Mask R-CNN. |
| **Mixed** | <img src="output/parameter_tuning/previews/mixed_preview.jpg" alt="residential" width="180" height="120"> | 4 | Strongest scene type. Mask R-CNN leads on recall and Dice; YOLO F1 is close because precision is higher. |
| **Tall** | <img src="output/parameter_tuning/previews/tall_preview.jpg" alt="residential" width="180" height="120"> | 4 | Only 16 labeled buildings total, so means are noisy. Mask R-CNN recall is ~1.0 but it over-detects. YOLO has better instance F1; Mask R-CNN still wins Dice except at 640 / 64. |
| **Wide** | <img src="output/parameter_tuning/previews/wide_preview.jpg" alt="residential" width="180" height="120"> | 4 | Do not pick tiling from this env. Instance detection fails; Mask R-CNN Dice is high because extra blobs still cover the large roofs. |
----

### Did we detect the right buildings (F1-Score)?

![F1-score per model per environment](output/parameter_tuning/f1_per_model_per_env.jpg)

![F1-score per model per tile size and overlap](output/parameter_tuning/f1_per_model_per_tilesize_per_overlap.jpg)

On **all** environments, mean F1 is similar (YOLO ~0.61, Mask R-CNN ~0.60) because YOLO’s precision offsets Mask R-CNN’s extra detections, and **wide** drags Mask R-CNN F1 down.

On **residential + mixed + tall only**, tile **640** is clearly best for both models. At 640 / 64:

| Model | Recall | Precision | F1 | Dice |
|-------|--------|-----------|-----|------|
| Mask R-CNN | **0.86** | 0.64 | 0.72 | **0.85** |
| YOLO | 0.70 | **0.82** | **0.74** | 0.79 |

YOLO’s headline F1 is slightly higher because it predicts fewer extras. Mask R-CNN’s recall is much higher (fewer missed buildings). That gap is what matters if the goal is to detect **all** buildings.

Per environment at **640 / 64**:

| Env | Model | Recall | Precision | F1 | Dice |
|-----|-------|--------|-----------|-----|------|
| Mixed | Mask R-CNN | **0.85** | 0.70 | **0.76** | **0.90** |
| Mixed | YOLO | 0.70 | 0.82 | 0.75 | 0.83 |
| Residential | Mask R-CNN | **0.74** | 0.64 | 0.67 | **0.89** |
| Residential | YOLO | 0.64 | 0.76 | **0.69** | 0.84 |
| Tall | Mask R-CNN | **1.00** | 0.58 | 0.73 | 0.77 |
| Tall | YOLO | 0.75 | **0.88** | **0.80** | 0.69 |

### Did the painted pixels overlap (Dice / IoU)?

![Dice score per model per environment](output/parameter_tuning/dice_per_model_per_env.jpg)

![Dice per model per tile size and overlap](output/parameter_tuning/dice_per_model_per_tilesize_per_overlap.jpg)

![IoU per model per tile size and overlap](output/parameter_tuning/iou_per_model_per_tilesize_per_overlap.jpg)

Dice/IoU are **not** the same ranking as F1. They measure coverage of the unioned masks. Extra instances that still sit on roofs **raise** Dice and **lower** F1.

On residential and mixed, Mask R-CNN Dice is consistently ~0.88–0.90 at 640 px, above YOLO (~0.77–0.84). That is the “exact footprint” signal: Mask R-CNN covers the roofs more completely.

Do not pick **overlap 192** from the Dice heatmap alone. Dice rises there because overlapping tiles add duplicate coverage; instance precision falls and inference time rises.

Tall at 640 / 64 is the one Mask R-CNN Dice dip (0.77 vs ~0.85 at overlap 32 or 128). With only 16 buildings, that cell is fragile. Overlap 32 keeps tall Dice high without hurting mixed.

### How long did it take to infer?

![Inference duration per model, tile size, and overlap](output/parameter_tuning/validation_duration_sec_per_model_per_tilesize_per_overlap.jpg)

Validation time is for the **full 16-image set** (same duration copied onto every row).

- **320 px** is slow (many tiles), especially overlap 192 (Mask R-CNN ~41 s, YOLO ~32 s), and F1 is worse.
- **640 px** is among the **fastest**: about **7.5–9.3 s**.
- YOLO is ~0.5–1 s faster than Mask R-CNN at the same 640 / 32–64 setting.
- Overlap 192 at 640 is only ~1 s slower than 64, but quality does not justify it for instance work.

### Recommendation

**Use fine-tuned Mask R-CNN, tile size 640, overlap 64.**

That is the best **single** setting for residential and mixed when both “find every building” (recall) and “match the roof outline” (Dice) matter, without a duration penalty.

- Mixed: best F1 (0.76) and Dice (0.90), recall 0.85.
- Residential: best Mask R-CNN F1 (0.67) and Dice (0.89), recall 0.74 vs YOLO’s 0.64.
- Inference ~8.3 s vs YOLO ~7.5 s on this validation set.
- Tile 640 / overlap 64 also ranked at the top of a joint score (harmonic mean of F1 and Dice) among Mask R-CNN cells, with the best building-weighted F1×Dice on the 12 target images.

**If tall footprints matter as much as mixed/residential:** prefer **640 / 32** (Mask R-CNN). Mixed stays strong (F1 0.75, Dice 0.90), tall Dice rises from 0.77 to 0.85, and inference is slightly faster (~8.0 s). Residential recall drops (0.67 vs 0.74).

**If false positives are more costly than misses:** YOLO **640 / 64** (or 640 / 32). Higher precision, slightly higher F1, worse recall and worse Dice.

Wide buildings need a different approach.
