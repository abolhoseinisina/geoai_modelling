import onnxruntime as ort
from ultralytics import YOLO
from common import ONNX_PATH, WEIGHTS_PATH

def execute() -> None:
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"{WEIGHTS_PATH} is missing - run generate_yolo.py first.")

    model = YOLO(str(WEIGHTS_PATH))
    exported = model.export(format="onnx", imgsz=512, opset=12, simplify=True, dynamic=False)
    print(f"exported {exported}")

    session = ort.InferenceSession(str(ONNX_PATH if ONNX_PATH.exists() else exported))
    print("inputs :", [i.name for i in session.get_inputs()])
    print("outputs:", [o.name for o in session.get_outputs()])

if __name__ == "__main__":
    execute()
