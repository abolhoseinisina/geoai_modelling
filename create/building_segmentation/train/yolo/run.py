import generate_data
import generate_onnx
import generate_yolo
import generate_tiles
from config import parseConfig

def main():
    config = parseConfig()
    print(f"Config: '{config.name}', device:{config.device}, epochs:{config.epochs}, batch:{config.batch_size}, model:{config.model}, imgsz={config.imgsz}")
    
    generate_tiles.execute()
    generate_data.execute()
    generate_yolo.execute(config)
    generate_onnx.execute()

if __name__ == "__main__":
    main()