import os
import boto3
from boto3 import Session
from botocore.client import Config
import tempfile
from contextlib import contextmanager
from django.conf import settings

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=settings.AWS_S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')  # Required for MinIO
    )

def upload_file_to_minio(file_obj, bucket_name, object_name):
    s3 = get_minio_client()
    s3.upload_fileobj(file_obj, bucket_name, object_name)
    return object_name

@contextmanager
def download_file_from_minio(bucket_name, object_name):
    s3 = get_minio_client()
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    try:
        s3.download_file(bucket_name, object_name, temp_path)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generate_presigned_url(object_key, expiration=3600):
    """
    Generates a temporary signed URL to view a file directly from MinIO.
    expiration: Time in seconds (default 1 hour).
    """
    s3 = get_minio_client()
    try:
        response = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return response
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None

def serialize_neo4j_node(node_data):
    if not node_data:
        return {}
    serialized_node = {}
    for key, value in node_data.items():
        if hasattr(value, 'isoformat'):
            serialized_node[key] = value.isoformat()
        elif isinstance(value, (list, dict)): # Podrías necesitar manejo recursivo para listas/dicts anidados
            serialized_node[key] = str(value) # o un manejo más específico
        else:
            serialized_node[key] = value
    return serialized_node
