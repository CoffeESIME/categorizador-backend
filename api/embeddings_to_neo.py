# embeddings_to_neo4j.py
from .neo4j_client import driver
import uuid
import os
import weaviate

# Conexión personalizada a Weaviate
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY", "")
    }
)
client.connect()
def close():
    driver.close()


def flatten_list_recursive(lst):
    """Aplana recursivamente una lista, eliminando anidamientos."""
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list_recursive(item))
        else:
            flat.append(item)
    return flat

def store_embedding(doc_id: str, embedding: list[float], meta: dict, label: str = "UnconnectedDoc"):
    """
    Almacena un nodo en Neo4j sin el embedding (solo los metadatos).
    El embedding se guardará en Weaviate por separado.
    """
    allowed_keys = [
        "author",
        "title",
        "work",
        "languages",
        "sentiment_word",
        "categories",
        "keywords",
        "content_type",
        "tags",
        "topics",
        "style",
        "file_location","analysis", "content"
    ]
    
    # Primero, hagamos un debug para ver qué estructura tienen los datos
    print("DEBUG - Meta data structure:")
    for key in meta:
        if key in allowed_keys:
            print(f"{key}: {type(meta[key])}")
            # Si es lista, verificar su contenido
            if isinstance(meta[key], list) and meta[key]:
                print(f"  First element type: {type(meta[key][0])}")
                # Si hay al menos un elemento que sea lista o diccionario
                if any(isinstance(elem, (list, dict)) for elem in meta[key]):
                    print(f"  WARNING: {key} contains nested collections!")
    
    # Preparar las propiedades validando cada una
    params = {"doc_id": doc_id, "id": str(uuid.uuid4())}
    
    # Procesar las propiedades una por una
    for key in allowed_keys:
        if key in meta:
            value = meta[key]
            
            # Convertir valores no compatibles a cadenas
            if isinstance(value, list):
                if any(isinstance(elem, (list, dict)) for elem in value):
                    print(f"Converting nested collection in {key} to string")
                    params[key] = str(value)  # Convertir a string
                else:
                    # Solo usar la lista si todos los elementos son primitivos
                    params[key] = value
            elif isinstance(value, dict):
                print(f"Converting dict in {key} to string")
                params[key] = str(value)
            else:
                params[key] = value
    
    # Construir la consulta
    set_clauses = []
    for key in params:
        set_clauses.append(f"n.{key} = ${key}")
    
    set_properties_cypher = ", ".join(set_clauses)
    
    # Ejecutar un comando UNWIND para los casos más complejos
    cypher = f"""
    CREATE (n:{label})
    SET {set_properties_cypher}
    RETURN n
    """
    
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            record = result.single()
            # También guardar en Weaviate si hay embedding
            if embedding and len(embedding) > 0:
                store_embedding_weaviate(doc_id, embedding, meta)
            return record["n"] if record else None
    except Exception as e:
        print(f"ERROR with Neo4j: {str(e)}")
        print(f"Parameters causing issues: {params}")
        # Intentar identificar cuál parámetro está causando el problema
        for key, value in params.items():
            print(f"Testing {key}...")
            test_params = {"test_value": value}
            try:
                with driver.session() as session:
                    session.run("RETURN $test_value", test_params)
                print(f"  {key} is OK")
            except Exception as param_error:
                print(f"  ERROR with {key}: {str(param_error)}")
        raise  # Re-lanzar la excepción original
      
    
def store_embedding_weaviate(doc_id: str, embedding: list[float], meta: dict):
    """
    Almacena el embedding junto con los metadatos en Weaviate.
    Selecciona la colección apropiada según el tipo de contenido.
    """
    # Determinar la colección de Weaviate según el tipo de contenido
    content_type = meta.get("content_type", "").lower()
    
    if content_type in ["image", "imagen", "photo", "foto"]:
        collection_name = "Imagenes"
    elif content_type in ["text", "texto", "article", "artículo", "book", "libro"]:
        collection_name = "Textos"
    elif content_type in ["audio", "sound", "música", "music"]:
        collection_name = "Audio"
    elif content_type in ["video", "película", "movie"]:
        collection_name = "Video"
    else:
        # Colección por defecto
        collection_name = "Textos"
    
    # Extraer propiedades relevantes del diccionario meta
    data_object = {
        "title": meta.get("title", "Sin título"),
        "doc_id": doc_id,
        "file_location": meta.get("file_location", ""),
        "analysis": meta.get("analysis", ""),
        "content": meta.get("content", "")
    }
    
    # Si tenemos autor, añadirlo al objeto
    if meta.get("author"):
        data_object["author"] = meta.get("author")
    
    # Crear colección si no existe
    try:
        collection = client.collections.get(collection_name)
    except Exception:
        print(f"La colección {collection_name} no existe, verificando esquema")
        return None
    
    try:
        # Guarda el objeto en Weaviate junto con su vector (embedding)
        result = client.collections.get(collection_name).data.insert(
            properties=data_object,
            vector=embedding
        )
        print(f"Objeto guardado en colección {collection_name} con ID: {result}")
        return result
    except Exception as e:
        print(f"Error al guardar embedding en Weaviate: {str(e)}")
        raise

def store_chunk_in_weaviate(client, chunk_text: str, embedding: list[float], original_doc_id: str, chunk_metadata: dict):
    """
    Almacena un chunk de texto y su embedding en Weaviate.
    Asocia el chunk con el ID del documento original.
    """
    collection_name = "PdfChunks"

    data_object = {
        "text_chunk": chunk_text,
        "original_doc_id": original_doc_id,
        "chunk_sequence": chunk_metadata.get("chunk_sequence", -1),
        "page_number": chunk_metadata.get("page_number", -1),
    }

    try:
        collection = client.collections.get(collection_name)
        uuid = collection.data.insert(
            properties=data_object,
            vector=embedding
        )
        print(f"Chunk de {original_doc_id} guardado en Weaviate ({collection_name}) con UUID: {uuid}")
        return uuid
    except Exception as e:
        print(f"Error guardando chunk en Weaviate para {original_doc_id}: {e}")
        return None
def guardar_imagen_en_weaviate(
    client,
    *,
    meta: dict,
    vec_clip: list[float] | None = None,   # vector visual
    vec_ocr:  list[float] | None = None,   # vector del OCR
    vec_desc: list[float] | None = None,   # vector de la descripción
) -> str | None:
    """
    Inserta un objeto en la colección 'Imagenes' con hasta tres
    named-vectors (vector_clip, vector_ocr, vector_des).

    — al menos uno de los tres debe estar presente —
    """
    if not any([vec_clip, vec_ocr, vec_desc]):
        raise ValueError("Proporciona al menos un vector")

    # 1. Limpia metadatos vacíos
    propiedades = {k: v for k, v in meta.items() if v is not None}

    # 2. Construye el diccionario de vectores con los que existan
    vectores = {}
    if vec_clip is not None:
        vectores["vector_clip"] = vec_clip
    if vec_ocr is not None:
        vectores["vector_ocr"] = vec_ocr
    if vec_desc is not None:
        vectores["vector_des"] = vec_desc

    # 3. Inserta en Weaviate
    imagenes = client.collections.get("Imagenes")
    uuid = imagenes.data.insert(properties=propiedades, vector=vectores)

    print(f"[OK] guardado {meta.get('doc_id')} · UUID: {uuid}")
    return uuid

def limpiar_meta(meta: dict) -> dict:
    """Devuelve un dict sin la clave prohibida 'id'."""
    meta = meta.copy()
    if "id" in meta:
        meta["doc_id"] = meta.pop("id")   # o el nombre que prefieras
    return {k: v for k, v in meta.items() if v is not None}