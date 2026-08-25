import os
import zipfile
import subprocess

S3_BUCKET = "s3://fm-ca-assets/geoai/"
ZIP_FILENAMES = ["building_segmentation_202608.zip"]

def downloadFromS3(zip_filename, zip_path):
    print(f"Downloading {zip_filename} from S3")
    result = subprocess.run(["aws", "s3", "cp", S3_BUCKET + zip_filename, zip_path], check=True)
    print("Download complete.")

def extractZip(zip_filename, zip_path):
    print(f"Extracting {zip_filename}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.getcwd())

    print("Extraction complete.")

def main():
    for zip_filename in ZIP_FILENAMES:
        zip_path = os.path.join(os.getcwd(), zip_filename)
        if not os.path.exists(zip_path):
            downloadFromS3(zip_filename, zip_path)
        
        else:
            print(f"{zip_filename} already exists. Skipping download.")
        
        extractZip(zip_filename, zip_path)

if __name__ == "__main__":
    main()