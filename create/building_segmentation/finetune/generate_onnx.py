import onnx
import torch
import warnings
import numpy as np
import onnxruntime as ort
from common import FINETUNED_PATH, ONNX_PATH, PRETRAINED_PATH, TILE_SIZE, buildModel

OPSET = 17
WEIGHTS_PATH = FINETUNED_PATH if FINETUNED_PATH.exists() else PRETRAINED_PATH

def export(model) -> None:
    dummy_input = [torch.rand(3, TILE_SIZE, TILE_SIZE)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (dummy_input,), str(ONNX_PATH), dynamo=False, export_params=True, opset_version=OPSET, do_constant_folding=True,
                          input_names=["images"], output_names=["boxes", "labels", "scores", "masks"], 
                          dynamic_axes={"images": {1: "height", 2: "width"}, "boxes": {0: "num_detections"}, "labels": {0: "num_detections"}, "scores": {0: "num_detections"}, "masks": {0: "num_detections"}})

    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print(f"exported {ONNX_PATH.name} (opset {OPSET})")

def verify(model) -> None:
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape) for i in session.get_inputs()])
    print("outputs:", [(o.name, o.shape) for o in session.get_outputs()])

    image = torch.rand(3, TILE_SIZE + 128, TILE_SIZE + 128)
    onnx_boxes, _, onnx_scores, _ = session.run(None, {"images": image.numpy()})
    with torch.no_grad():
        torch_output = model([image])[0]

    torch_boxes = torch_output["boxes"].numpy()
    print(f"detections - onnx: {len(onnx_boxes)}  torch: {len(torch_boxes)}")
    if len(onnx_boxes) == len(torch_boxes) and len(onnx_boxes):
        print(f"max abs box diff: {np.abs(onnx_boxes - torch_boxes).max():.4f}  max abs score diff: {np.abs(onnx_scores - torch_output['scores'].numpy()).max():.6f}")

def execute() -> None:
    if WEIGHTS_PATH is PRETRAINED_PATH:
        print(f"{FINETUNED_PATH.name} not found")
        exit()
    
    print(f"loading {WEIGHTS_PATH.name}")
    model = buildModel(weights_path=WEIGHTS_PATH)
    model.eval()
    export(model)
    verify(model)

if __name__ == "__main__":
    execute()