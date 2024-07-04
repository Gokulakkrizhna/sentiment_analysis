import boto3
import os
import subprocess

# AWS S3 configuration
s3_bucket = 'please provide bucket name'

# AWS credentials
aws_access_key_id = 'please provide s3 access id'
aws_secret_access_key = 'please provide s3 access key'

# Files to download from S3
files_to_download = {
    'app.py': '/home/ec2-user/app.py',
    'knn.pkl': '/home/ec2-user/knn.pkl',
    'vectorizer.pkl': '/home/ec2-user/vectorizer.pkl'
}

# Initialize S3 client with credentials
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Download the files from S3
for s3_key, local_path in files_to_download.items():
    s3.download_file(s3_bucket, s3_key, local_path)

# Run the Streamlit app
subprocess.run(['streamlit', 'run', '/home/ec2-user/app.py'])
