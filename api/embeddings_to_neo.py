# embeddings_to_neo4j.py
from .neo4j_client import driver
import uuid
import os
import weaviate
from django.conf import settings

# Conexión personalizada a Weaviate
client = weaviate.connect_to_custom(
    http_host=settings.WEAVIATE_HTTP_HOST,
    http_port=settings.WEAVIATE_HTTP_PORT,
    http_secure=settings.WEAVIATE_HTTP_SECURE,
    grpc_host=settings.WEAVIATE_GRPC_HOST,
    grpc_port=settings.WEAVIATE_GRPC_PORT,
    grpc_secure=settings.WEAVIATE_GRPC_SECURE,
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

def store_embedding(doc_id: str, embedding: list[float], meta: dict, label: str = "DigitalAsset"):
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
    
    # Construir etiquetas dinámicas
    labels = [label, "Inbox"]
    
    # Añadir etiqueta de content_type si existe
    if "content_type" in meta and meta["content_type"]:
        ct = meta["content_type"]
        # Simple sanitización para asegurar que sea un string válido para etiqueta
        if isinstance(ct, str) and ct.isalnum():
            labels.append(ct.capitalize())
            
    labels_str = ":".join(labels)
    
    # Ejecutar un comando UNWIND para los casos más complejos
    cypher = f"""
    CREATE (n:{labels_str})
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
    vec_clip: list[float] | None = None,
    vec_ocr:  list[float] | None = None,
    vec_desc: list[float] | None = None,
) -> str | None:
    
    propiedades_for_weaviate = {k: v for k, v in meta.items() if v is not None}

    if "doc_id" in propiedades_for_weaviate:
        propiedades_for_weaviate["doc_id"] = str(propiedades_for_weaviate["doc_id"])
    else:
        print(f"ADVERTENCIA: 'doc_id' no encontrado en meta para guardar_imagen_en_weaviate. Meta: {meta}")

    vectores = {}
    if vec_clip is not None:
        vectores["vector_clip"] = vec_clip
    if vec_ocr is not None:
        vectores["vector_ocr"] = vec_ocr
    if vec_desc is not None:
        vectores["vector_des"] = vec_desc

    try:
        imagenes_collection = client.collections.get("Imagenes")
        
        print(f"DEBUG: Enviando a Weaviate (Imagenes) propiedades: {propiedades_for_weaviate}")

        if not vectores:
            insert_result_uuid = imagenes_collection.data.insert(properties=propiedades_for_weaviate)
        else:
            insert_result_uuid = imagenes_collection.data.insert(properties=propiedades_for_weaviate, vector=vectores)
        
        # CORRECCIÓN AQUÍ:
        # insert_result_uuid ya es el objeto uuid.UUID directamente.
        # No necesitas acceder a .uuid en él.
        
        print(f"[OK] Objeto guardado en Imagenes {propiedades_for_weaviate.get('doc_id')} · UUID: {insert_result_uuid}")
        return str(insert_result_uuid) # Convertir el objeto uuid.UUID a string para el retorno
        
    except Exception as e:
        print(f"Error al guardar en Weaviate (Imagenes) para doc_id {propiedades_for_weaviate.get('doc_id')}: {e}")
        print(f"DEBUG: Propiedades que causaron error en Weaviate: {propiedades_for_weaviate}")
        raise

def guardar_video_en_weaviate(
    client,
    *,
    meta: dict,
    vec_video: list[float] | None = None,
    vec_audio: list[float] | None = None,
    vec_text: list[float] | None = None,
) -> str | None:

    propiedades_for_weaviate = {k: v for k, v in meta.items() if v is not None}

    if "doc_id" in propiedades_for_weaviate:
        propiedades_for_weaviate["doc_id"] = str(propiedades_for_weaviate["doc_id"])
    else:
        print(
            f"ADVERTENCIA: 'doc_id' no encontrado en meta para guardar_video_en_weaviate. Meta: {meta}"
        )

    vectores = {}
    if vec_video is not None:
        vectores["vector_video"] = vec_video
    if vec_audio is not None:
        vectores["vector_audio"] = vec_audio
    if vec_text is not None:
        vectores["vector_text"] = vec_text

    try:
        videos_collection = client.collections.get("Video")

        print(
            f"DEBUG: Enviando a Weaviate (Video) propiedades: {propiedades_for_weaviate}"
        )

        if not vectores:
            insert_result_uuid = videos_collection.data.insert(properties=propiedades_for_weaviate)
        else:
            insert_result_uuid = videos_collection.data.insert(
                properties=propiedades_for_weaviate, vector=vectores
            )

        print(
            f"[OK] Objeto guardado en Video {propiedades_for_weaviate.get('doc_id')} · UUID: {insert_result_uuid}"
        )
        return str(insert_result_uuid)

    except Exception as e:
        print(
            f"Error al guardar en Weaviate (Video) para doc_id {propiedades_for_weaviate.get('doc_id')}: {e}"
        )
        print(
            f"DEBUG: Propiedades que causaron error en Weaviate: {propiedades_for_weaviate}"
        )
        raise



def guardar_audio_en_weaviate(
    client,
    *,
    meta: dict,
    vec_audio: list[float] | None = None,
    vec_text: list[float] | None = None,
) -> str | None:

    propiedades_for_weaviate = {k: v for k, v in meta.items() if v is not None}

    if "doc_id" in propiedades_for_weaviate:
        propiedades_for_weaviate["doc_id"] = str(propiedades_for_weaviate["doc_id"])
    else:
        print(
            f"ADVERTENCIA: 'doc_id' no encontrado en meta para guardar_audio_en_weaviate. Meta: {meta}"
        )

    vectores = {}
    if vec_audio is not None:
        vectores["vector_audio"] = vec_audio
    if vec_text is not None:
        vectores["vector_text"] = vec_text

    try:
        audio_collection = client.collections.get("Audio")

        print(
            f"DEBUG: Enviando a Weaviate (Audio) propiedades: {propiedades_for_weaviate}"
        )

        if not vectores:
            insert_result_uuid = audio_collection.data.insert(properties=propiedades_for_weaviate)
        else:
            insert_result_uuid = audio_collection.data.insert(
                properties=propiedades_for_weaviate, vector=vectores
            )

        print(
            f"[OK] Objeto guardado en Audio {propiedades_for_weaviate.get('doc_id')} · UUID: {insert_result_uuid}"
        )
        return str(insert_result_uuid)

    except Exception as e:
        print(
            f"Error al guardar en Weaviate (Audio) para doc_id {propiedades_for_weaviate.get('doc_id')}: {e}"
        )
        print(
            f"DEBUG: Propiedades que causaron error en Weaviate: {propiedades_for_weaviate}"
        )
        raise


def limpiar_meta(meta: dict) -> dict:
    meta_copy = meta.copy()
    if "id" in meta_copy:
        # Asegurar que el valor que se convierte en doc_id sea una cadena
        meta_copy["doc_id"] = str(meta_copy.pop("id"))
    # Si 'doc_id' ya existe y no hay 'id', asegurarse de que 'doc_id' también sea string
    elif "doc_id" in meta_copy:
         meta_copy["doc_id"] = str(meta_copy["doc_id"])
    return {k: v for k, v in meta_copy.items() if v is not None}