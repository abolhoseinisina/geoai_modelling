import json
import math
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from BuildingTileDataset import BuildingTileDataset
from common import FINETUNED_PATH, PRETRAINED_PATH, TILE_INDEX, buildModel, freezeBackbone, freezeBatchnorm, getDevice

EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.002  # SGD, scaled down from the 0.02/batch-16 reference recipe
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 10.0
VAL_FRACTION = 0.15
SEED = 42
DEVICE = "auto"
TRAINABLE_BACKBONE_STAGES = 3
NUM_WORKERS = 0

def splitTrainingValidationRecords(records: list[dict]) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(SEED).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * VAL_FRACTION))
    return shuffled[cut:], shuffled[:cut]

def collate(batch):
    return tuple(zip(*batch))

def getLRSchedule(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return 0.05 + 0.95 * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor

@torch.no_grad()
def getValidationLoss(model, loader, device) -> dict[str, float]:
    model.train()
    freezeBatchnorm(model)
    totals: dict[str, float] = {}
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in t.items()} for t in targets]
        for key, value in model(images, targets).items():
            totals[key] = totals.get(key, 0.0) + value.item()
    
    return {key: value / max(len(loader), 1) for key, value in totals.items()}

@torch.no_grad()
def getModelPerformance(model, loader, device, score_thresh=0.5, iou_thresh=0.5) -> tuple[float, float]:
    model.eval()
    true_positives = false_positives = ground_truths = 0
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        for prediction, target in zip(predictions, targets):
            keep = prediction["scores"] > score_thresh
            predicted = (prediction["masks"][keep, 0] > 0.5).flatten(1).float().cpu()
            truth = target["masks"].bool().flatten(1).float()
            ground_truths += len(truth)
            if len(predicted) == 0 or len(truth) == 0:
                false_positives += len(predicted)
                continue
            intersection = truth @ predicted.T
            union = truth.sum(1, keepdim=True) + predicted.sum(1) - intersection
            best = (intersection / union.clamp(min=1)).max(dim=1).values
            matched = int((best > iou_thresh).sum())
            true_positives += matched
            false_positives += max(len(predicted) - matched, 0)
    
    recall = true_positives / max(ground_truths, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    return recall, precision

def execute() -> None:
    print('Start fine-tuning')
    if not TILE_INDEX.exists():
        raise SystemExit(f"{TILE_INDEX} is missing - run generate_tiles.py first.")

    torch.manual_seed(SEED)
    random.seed(SEED)

    records = json.loads(TILE_INDEX.read_text())
    train_records, val_records = splitTrainingValidationRecords(records)
    train_neg = sum(1 for r in train_records if not r["polygons"])
    val_neg = sum(1 for r in val_records if not r["polygons"])
    print(f"{len(train_records)} training tiles ({train_neg} empty), {len(val_records)} validation tiles ({val_neg} empty)")
    
    train_loader = DataLoader(BuildingTileDataset(train_records, augment=True), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(BuildingTileDataset(val_records, augment=False), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=NUM_WORKERS)

    device = getDevice(DEVICE)
    print(f"Device: {device}")

    model = buildModel(weights_path=PRETRAINED_PATH)
    freezeBackbone(model, TRAINABLE_BACKBONE_STAGES)
    freezeBatchnorm(model)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, getLRSchedule(total_steps, warmup_steps=min(len(train_loader), 100)))

    print('Get pretrained model performance')
    baseline = getModelPerformance(model, val_loader, device)
    print(f" recall: {baseline[0]:.3f} - precision: {baseline[1]:.3f}")
    
    best_val = math.inf
    epoch_tqdm = tqdm(range(EPOCHS), desc='Fine-tuning', ncols=100)
    for _ in epoch_tqdm:
        model.train()
        freezeBatchnorm(model)

        running, batches = 0.0, 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("  skipping batch with non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            batches += 1

        train_loss = running / max(batches, 1)
        val_losses = getValidationLoss(model, val_loader, device)
        val_loss = sum(val_losses.values())
        epoch_tqdm.set_postfix_str(f"train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), FINETUNED_PATH)

    model.load_state_dict(torch.load(FINETUNED_PATH, map_location=device, weights_only=True))
    final = getModelPerformance(model, val_loader, device)
    print('Get fine-tuned model performance')
    print(f'recall: {final[0]:.3f} - precision: {final[1]:.3f}')
    print(f'weights     : {FINETUNED_PATH}')
    print(f"Fine-tuned model saved {FINETUNED_PATH.name} (best val {best_val:.4f})")

if __name__ == "__main__":
    execute()