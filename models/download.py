import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

load_dotenv()

S3_BUCKET = 'fm-ca-assets'
S3_PREFIX = 'geoai/models'
MODEL_FILENAMES = ['building_footprints_usa.pth']

def downloadFromS3(filename, file_path):
    s3 = boto3.client('s3')
    key = f'{S3_PREFIX}/{filename}'
    try:
        print(f'Downloading "{filename}" from S3 bucket "{S3_BUCKET}"')
        s3.download_file(S3_BUCKET, key, file_path)
        print('Download complete.')

    except NoCredentialsError:
        print('AWS credentials not found. Please configure your AWS credentials.')
        raise
    
    except ClientError as e:
        print(f'Error downloading file from S3: {e}')
        raise

def main():
    for model_filename in MODEL_FILENAMES:
        model_path = os.path.join(os.getcwd(), model_filename)
        if os.path.exists(model_path):
            print(f'{model_filename} already exists. Skipping download.')
            return

        downloadFromS3(model_filename, model_path)

if __name__ == '__main__':
    main()