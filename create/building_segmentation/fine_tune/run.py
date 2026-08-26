import generate_tiles
import generate_onnx
import compare_models
from config import parseConfig
import generate_finetuned_model

def main():
    config = parseConfig()
    print(f"Config: '{config.name}', device:{config.device}, epochs:{config.epochs}, batch:{config.batch_size}, lr:{config.learning_rate}")
    
    generate_tiles.execute()
    generate_finetuned_model.execute(config)
    generate_onnx.execute()
    compare_models.execute(config)

if __name__ == '__main__':
    main()