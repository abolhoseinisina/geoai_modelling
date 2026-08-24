import generate_tiles
import generate_finetuned_model
import generate_onnx
import compare_models

def main():
    generate_tiles.execute()
    generate_finetuned_model.execute()
    generate_onnx.execute()
    compare_models.execute()

if __name__ == '__main__':
    main()