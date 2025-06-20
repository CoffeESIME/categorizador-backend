# weaviate_client.py
import os
import weaviate
from django.conf import settings

def get_client():
    client = weaviate.connect_to_custom(
        http_host=settings.WEAVIATE_HTTP_HOST,
        http_port=settings.WEAVIATE_HTTP_PORT,
        http_secure=settings.WEAVIATE_HTTP_SECURE,
        grpc_host=settings.WEAVIATE_GRPC_HOST,
        grpc_port=settings.WEAVIATE_GRPC_PORT,
        grpc_secure=settings.WEAVIATE_GRPC_SECURE,
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY", "")},
    )
    client.connect()
    return client

CLIENT = get_client()
