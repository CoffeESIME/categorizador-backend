import weaviate
from neo4j import GraphDatabase
import boto3
from botocore.exceptions import ClientError

# --- CONFIGURACIÓN ---
NAS_IP = "192.168.50.2"  # <--- PON TU IP AQUÍ
MINIO_PASS = "mi_password_seguro"
NEO4J_PASS = "mi_password_seguro"

print(f"🚀 Iniciando diagnóstico del NAS en {NAS_IP}...\n")

# 1. PRUEBA WEAVIATE (Versión 4 - Corregida)
try:
    print("🔹 Probando Weaviate (Vectores)...", end=" ")
    
    # Conexión explícita para v4 apuntando a tu NAS
    client = weaviate.connect_to_custom(
        http_host=NAS_IP,
        http_port=8081,
        http_secure=False,
        grpc_host=NAS_IP,
        grpc_port=50051,
        grpc_secure=False,
    )
    
    try:
        # Intentamos obtener la metadata para verificar conexión
        meta = client.get_meta()
        print(f"✅ OK! Conectado a Weaviate v{meta['version']}")
    finally:
        # Siempre hay que cerrar el cliente en v4
        client.close()

except Exception as e:
    print(f"❌ FALLÓ: {e}")

# 2. PRUEBA NEO4J
try:
    print("🔹 Probando Neo4j (Grafos)...", end=" ")
    uri = f"bolt://{NAS_IP}:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", NEO4J_PASS))
    with driver.session() as session:
        res = session.run("RETURN 'Conexión Exitosa' AS mensaje")
        msg = res.single()["mensaje"]
    driver.close()
    print(f"✅ OK! Mensaje recibido: {msg}")
except Exception as e:
    print(f"❌ FALLÓ: {e}")

# 3. PRUEBA MINIO (S3)
try:
    print("🔹 Probando MinIO (Archivos)...", end=" ")
    s3 = boto3.client('s3',
                      endpoint_url=f"http://{NAS_IP}:9000",
                      aws_access_key_id="admin",
                      aws_secret_access_key=MINIO_PASS,
                      config=boto3.session.Config(signature_version='s3v4'))
    
    response = s3.list_buckets()
    buckets = [b['Name'] for b in response['Buckets']]
    print(f"✅ OK! Buckets encontrados: {buckets}")
except Exception as e:
    print(f"❌ FALLÓ: {e}")

print("\n🏁 Diagnóstico finalizado.")