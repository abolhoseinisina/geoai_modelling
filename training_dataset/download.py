import os
import boto3
import zipfile
from botocore.exceptions import NoCredentialsError, ClientError

S3_BUCKET = 'fm-ca-assets'
S3_PREFIX = 'geoai'
ZIP_FILENAMES = ['building_segmentation_202608.zip']

def downloadFromS3(zip_filename, zip_path):
    s3 = boto3.client('s3')
    key = f'{S3_PREFIX}/{zip_filename}'
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

def extractZip(zip_filename, zip_path):
    print(f'Extracting "{zip_filename}"')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

    print('Extraction complete.')

def main():
    for zip_filename in ZIP_FILENAMES:
        zip_path = os.path.join(os.getcwd(), zip_filename)
        if not os.path.exists(zip_path):
            downloadFromS3(zip_filename, zip_path)
        
        else:
            print(f'{zip_filename} already exists. Skipping download.')
        
        extractZip(zip_filename, zip_path)

if __name__ == '__main__':
    main()