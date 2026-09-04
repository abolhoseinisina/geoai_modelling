import os
import boto3
import zipfile
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

load_dotenv()

S3_BUCKET = 'fm-ca-assets'
S3_PREFIX = 'geoai/building_segmentation'
DATASET_REPOS = {
    'training_datasets': 'building_segmentation_202608.zip',
    'validating_datasets': 'building_segmentation_202608.zip'
}

def downloadFromS3(repo, zip_filename, zip_path):
    s3 = boto3.client('s3')
    key = f'{S3_PREFIX}/{repo}/{zip_filename}'
    print(key)
    try:
        print(f'Downloading "{zip_filename}" from S3 bucket "{S3_BUCKET}"')
        s3.download_file(S3_BUCKET, key, zip_path)
        print('Download complete.')

    except NoCredentialsError:
        print('AWS credentials not found. Please configure your AWS credentials.')
        raise
    
    except ClientError as e:
        print(f'Error downloading file from S3: {e}')
        raise

def extractZip(zip_filename, zip_path, dataset_repo):
    print(f'Extracting "{zip_filename}"')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd() + '/' + dataset_repo)

    print('Extraction complete.')

def main():
    for dataset_repo, dataset_file in DATASET_REPOS.items():
        zip_path = os.path.join(os.getcwd(), dataset_repo + '/' + dataset_file)
        if not os.path.exists(zip_path):
            os.makedirs(dataset_repo, exist_ok=True)
            downloadFromS3(dataset_repo, dataset_file, zip_path)
        
        else:
            print(f'{dataset_file} already exists. Skipping download.')
        
        extractZip(dataset_file, zip_path, dataset_repo)

if __name__ == '__main__':
    main()