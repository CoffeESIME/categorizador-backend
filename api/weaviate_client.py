# weaviate_client.py
import os, weaviate

def get_client():
    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY", "")},
    )
    client.connect()
    return client

CLIENT = get_client()
