import os
import json
import boto3
from botocore.exceptions import ClientError

from api.neo4j_client import driver
import weaviate
import weaviate.classes.config as wc
from django.conf import settings


def seed_node_types():
    """
    Define el esquema de metadatos (Ontología) en Neo4j.
    Estrategia Híbrida: Estructura Core + Flexibilidad Reactiva.
    """
    node_types = [
        # ---------------------------------------------------------
        # NIVEL 1: SOPORTE DE ARCHIVOS (Datos Técnicos + Flexibilidad)
        # ---------------------------------------------------------
        {
            "id": "digital_asset",
            "name": "Digital Asset (Archivo Base)",
            "description": "Datos técnicos inmutables y contexto personal flexible.",
            "fields": json.dumps([
                # --- Core Técnico ---
                {"fieldName": "doc_id", "placeholder": "UUID del sistema", "required": True, "readonly": True},
                {"fieldName": "file_hash", "placeholder": "SHA256 Hash (Deduplicación)", "required": True, "readonly": True},
                {"fieldName": "file_path", "placeholder": "Ruta relativa", "required": True, "readonly": True},
                {"fieldName": "mime_type", "placeholder": "MIME Type", "required": True, "readonly": True},
                {"fieldName": "size_bytes", "placeholder": "Tamaño (bytes)", "required": True, "readonly": True},
                {"fieldName": "original_name", "placeholder": "Nombre original", "required": True},
                {"fieldName": "creation_date", "placeholder": "Fecha de creación", "required": False},

                # --- FLEXIBILIDAD TOTAL (Catch-All) ---
                # Aquí guardas anécdotas: "Me la recomendó Juan en la fiesta"
                {"fieldName": "user_memories", "placeholder": "Memorias / Contexto personal", "required": False, "widget": "textarea"},
                # Aquí guardas datos técnicos raros: {"bpm": 120, "iso": 400}
                {"fieldName": "dynamic_metadata", "placeholder": "Metadatos extra (JSON)", "required": False, "widget": "json_editor"},
            ])
        },
        
        # ---------------------------------------------------------
        # NIVEL 2: INBOX / STAGING (Sugerencias de IA para Curaduría)
        # Estos campos son temporales. El usuario los edita y al guardar se convierten en relaciones.
        # ---------------------------------------------------------
        {
            "id": "inbox_item",
            "name": "Inbox / Metadatos Sugeridos",
            "description": "Espacio de trabajo para revisión humana antes de la conexión.",
            "fields": json.dumps([
                # --- Estado del Proceso ---
                {"fieldName": "processing_status", "placeholder": "pending/done/error", "required": True},
                {"fieldName": "error_log", "placeholder": "Log de errores", "readonly": True, "widget": "textarea"},
                
                # --- Análisis Semántico General ---
                {"fieldName": "ai_summary", "placeholder": "Resumen IA", "readonly": True, "widget": "textarea"},
                {"fieldName": "ai_insights", "placeholder": "Insights extra detectados", "readonly": True, "widget": "textarea"},
                {"fieldName": "detected_language", "placeholder": "Idioma detectado (es, en)", "required": False},
                
                # --- Input del Usuario al Vuelo ---
                {"fieldName": "user_notes_input", "placeholder": "Nota rápida al subir", "required": False, "widget": "textarea"},

                # --- Dimensiones Artísticas y Estructurales (Sugeridas) ---
                {"fieldName": "suggested_title", "placeholder": "Título sugerido", "required": False},
                {"fieldName": "suggested_style", "placeholder": "Estilo (Barroco, Cyberpunk)", "required": False},
                {"fieldName": "suggested_technique", "placeholder": "Técnica/Forma (Óleo, Haiku)", "required": False},
                {"fieldName": "suggested_mood", "placeholder": "Atmósfera/Mood", "required": False},

                # --- Relaciones Sugeridas (Texto plano -> Futura Conexión) ---
                {"fieldName": "suggested_author", "placeholder": "Autor/Creador detectado", "required": False},
                {"fieldName": "suggested_tags", "placeholder": "Tags (lista)", "required": False},
                {"fieldName": "suggested_topics", "placeholder": "Temas (Conceptos)", "required": False},
                {"fieldName": "suggested_location", "placeholder": "Lugar/País mencionado", "required": False},
                {"fieldName": "related_work", "placeholder": "Obra/Proyecto relacionado", "required": False},
                
                # --- Sentimiento ---
                {"fieldName": "sentiment_label", "placeholder": "Sentimiento (Texto)", "required": False},
                {"fieldName": "sentiment_score", "placeholder": "Sentimiento (Numérico -1 a 1)", "required": False, "type": "number"},
                
                # --- Análisis Profundo ---
                {"fieldName": "analysis", "placeholder": "Análisis detallado", "required": False, "widget": "textarea"},
            ])
        },

        # ---------------------------------------------------------
        # NIVEL 3: TIPOS DE CONTENIDO (Propiedades Específicas)
        # ---------------------------------------------------------
        {
            "id": "audio",
            "name": "Audio / Música",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título", "required": True},
                {"fieldName": "artist", "placeholder": "Artista", "required": False},
                {"fieldName": "album", "placeholder": "Álbum", "required": False},
                {"fieldName": "release_year", "placeholder": "Año", "required": False},
                {"fieldName": "genre", "placeholder": "Género", "required": False},
                {"fieldName": "lyrics", "placeholder": "Letra / Transcripción", "required": False, "widget": "textarea"},
                {"fieldName": "duration", "placeholder": "Duración (s)", "required": False},
            ])
        },
        {
            "id": "image",
            "name": "Imagen / Arte",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título", "required": True},
                {"fieldName": "visual_style", "placeholder": "Estilo Visual", "required": False},
                {"fieldName": "technique", "placeholder": "Técnica", "required": False},
                {"fieldName": "description", "placeholder": "Descripción Visual (Caption)", "required": False, "widget": "textarea"},
                {"fieldName": "ocr_text", "placeholder": "Texto en imagen (OCR)", "required": False, "widget": "textarea"},
                {"fieldName": "resolution", "placeholder": "Resolución", "required": False},
                {"fieldName": "camera_model", "placeholder": "Cámara", "required": False},
                {"fieldName": "taken_at", "placeholder": "Fecha Captura", "required": False},
            ])
        },
        {
            "id": "video",
            "name": "Video",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título", "required": True},
                {"fieldName": "cinematography", "placeholder": "Estilo Visual / Cine", "required": False},
                {"fieldName": "transcript", "placeholder": "Transcripción (Voz)", "required": False, "widget": "textarea"},
                {"fieldName": "duration", "placeholder": "Duración (s)", "required": False},
                {"fieldName": "framerate", "placeholder": "FPS", "required": False},
            ])
        },
        {
            "id": "document",
            "name": "Documento (Texto)",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título", "required": True},
                {"fieldName": "summary", "placeholder": "Resumen", "required": False, "widget": "textarea"},
                {"fieldName": "language", "placeholder": "Idioma", "required": False},
                {"fieldName": "page_count", "placeholder": "Páginas", "required": False},
                {"fieldName": "type", "placeholder": "Tipo Doc (Ensayo, Noticia)", "required": False},
            ])
        },
        {
            "id": "quote",
            "name": "Cita / Poema Corto",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título (si aplica)", "required": False},
                {"fieldName": "text", "placeholder": "Texto / Versos", "required": True, "widget": "textarea"},
                {"fieldName": "form", "placeholder": "Forma (Haiku, Soneto)", "required": False},
                {"fieldName": "author_ref", "placeholder": "Autor (Referencia)", "required": False},
                {"fieldName": "context", "placeholder": "Contexto original", "required": False},
                {"fieldName": "sentiment_score", "placeholder": "Score", "required": False, "type": "number"},
            ])
        },
        {
            "id": "post",
            "name": "Publicación (Post)",
            "fields": json.dumps([
                {"fieldName": "content", "placeholder": "Contenido", "required": True, "widget": "textarea"},
                {"fieldName": "platform", "placeholder": "Plataforma", "required": False},
                {"fieldName": "url", "placeholder": "URL original", "required": False},
                {"fieldName": "metrics", "placeholder": "Métricas (JSON)", "required": False},
            ])
        },

        # ---------------------------------------------------------
        # NIVEL 4: ENTIDADES DE CONOCIMIENTO (Destino de Conexiones)
        # ---------------------------------------------------------
        {
            "id": "person",
            "name": "Persona",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre completo", "required": True},
                {"fieldName": "roles", "placeholder": "Roles (Autor, Músico)", "required": False},
                {"fieldName": "bio", "placeholder": "Biografía", "required": False},
                {"fieldName": "birthdate", "placeholder": "Fecha nacimiento", "required": False},
            ])
        },
        {
            "id": "location",
            "name": "Ubicación / Lugar",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del lugar", "required": True},
                {"fieldName": "type", "placeholder": "Tipo (Ciudad, País, Spot)", "required": False},
                {"fieldName": "coords", "placeholder": "Lat,Lon", "required": False},
                {"fieldName": "iso_code", "placeholder": "ISO Code", "required": False},
            ])
        },
        {
            "id": "concept",
            "name": "Concepto / Estilo",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre", "required": True},
                {"fieldName": "type", "placeholder": "Tipo (Estilo, Tema, Técnica)", "required": False},
                {"fieldName": "definition", "placeholder": "Definición", "required": False},
                {"fieldName": "domain", "placeholder": "Dominio (Arte, Ciencia)", "required": False},
            ])
        },
        {
            "id": "project",
            "name": "Proyecto / Obra",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título", "required": True},
                {"fieldName": "type", "placeholder": "Tipo (Libro, Álbum)", "required": False},
                {"fieldName": "year", "placeholder": "Año", "required": False},
            ])
        },
        {
            "id": "organization",
            "name": "Organización",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre", "required": True},
                {"fieldName": "industry", "placeholder": "Industria", "required": False},
                {"fieldName": "website", "placeholder": "Sitio Web", "required": False},
            ])
        },
        {
            "id": "event",
            "name": "Evento",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre", "required": True},
                {"fieldName": "date", "placeholder": "Fecha", "required": False},
                {"fieldName": "type", "placeholder": "Tipo", "required": False},
            ])
        },
        {
            "id": "tag",
            "name": "Etiqueta",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Tag", "required": True},
            ])
        },
        {
            "id": "source",
            "name": "Fuente Externa",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título", "required": True},
                {"fieldName": "url", "placeholder": "URL", "required": False},
            ])
        }
    ]

    with driver.session() as session:
        for node_type in node_types:
            query = """
            MERGE (nt:NodeType {id: $id})
            SET nt.name = $name,
                nt.description = $description,
                nt.fields = $fields
            """
            session.run(
                query,
                id=node_type["id"],
                name=node_type["name"],
                description=node_type.get("description", ""),
                fields=node_type["fields"]
            )

    print("✅ Tipos de Nodo (Ontología Neo4j) actualizados con éxito.")


def ensure_minio_structure():
    """
    Inicializa el Bucket y las 'carpetas' base en MinIO.
    """
    print(f"🌊 Conectando a MinIO en {settings.AWS_S3_ENDPOINT_URL}...")

    s3 = boto3.client(
        's3',
        endpoint_url=settings.AWS_S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    # 1. Crear el Bucket si no existe
    try:
        s3.head_bucket(Bucket=settings.AWS_STORAGE_BUCKET_NAME)
        print(f"✔ Bucket '{settings.AWS_STORAGE_BUCKET_NAME}' ya existe.")
    except ClientError:
        print(f"✨ Creando bucket '{settings.AWS_STORAGE_BUCKET_NAME}'...")
        s3.create_bucket(Bucket=settings.AWS_STORAGE_BUCKET_NAME)

    # 2. Crear estructura de carpetas (Simulación S3)
    # En S3 las carpetas son objetos vacíos que terminan en '/'
    subdirs = ['images/', 'videos/', 'audio/', 'documents/', 'texts/', 'others/']
    
    for folder in subdirs:
        try:
            s3.put_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=folder)
            print(f"  📂 Carpeta '{folder}' verificada/creada.")
        except Exception as e:
            print(f"  ❌ Error creando '{folder}': {e}")

    print("✅ Estructura MinIO sincronizada.")


def seed_weaviate_schema() -> None:
    """
    Configura el esquema de Weaviate con MULTI-VECTOR RETRIEVER y soporte para CONTEXTO DE USUARIO.
    Define múltiples espacios vectoriales para cada tipo de contenido.
    """
    client = weaviate.connect_to_custom(
        http_host=settings.WEAVIATE_HTTP_HOST,
        http_port=settings.WEAVIATE_HTTP_PORT,
        http_secure=settings.WEAVIATE_HTTP_SECURE,
        grpc_host=settings.WEAVIATE_GRPC_HOST,
        grpc_port=settings.WEAVIATE_GRPC_PORT,
        grpc_secure=settings.WEAVIATE_GRPC_SECURE,
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY", "")}
    )
    client.connect()

    # Propiedades base para el enlace con Neo4j
    common_props = [
        wc.Property(name="doc_id", data_type=wc.DataType.TEXT, tokenization=wc.Tokenization.FIELD),
        wc.Property(name="file_hash", data_type=wc.DataType.TEXT, tokenization=wc.Tokenization.FIELD),
        wc.Property(name="file_location", data_type=wc.DataType.TEXT),
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="tags", data_type=wc.DataType.TEXT_ARRAY),
        # Campo para el texto plano de las memorias del usuario
        wc.Property(name="user_memories", data_type=wc.DataType.TEXT), 
    ]

    collections_cfg = [
        dict(
            name="Image",
            description="Imágenes con multi-vector (Visual, OCR, Semántico, Usuario)",
            vectorizer_config=[
                wc.Configure.NamedVectors.none(name="visual"),       # CLIP (Imagen)
                wc.Configure.NamedVectors.none(name="ocr"),          # Texto OCR
                wc.Configure.NamedVectors.none(name="description"),  # Caption Semántico
                wc.Configure.NamedVectors.none(name="user_context"), # Vector de Memorias del Usuario
            ],
            properties=common_props + [
                wc.Property(name="ocr_text", data_type=wc.DataType.TEXT),
                wc.Property(name="description", data_type=wc.DataType.TEXT),
            ],
        ),
        dict(
            name="Video",
            description="Videos con multi-vector (Visual, Audio, Transcript, Usuario)",
            vectorizer_config=[
                wc.Configure.NamedVectors.none(name="visual"),      # XCLIP (Frames)
                wc.Configure.NamedVectors.none(name="audio"),       # AudioCLIP (Sonido/Música)
                wc.Configure.NamedVectors.none(name="transcript"),  # Whisper (Voz)
                wc.Configure.NamedVectors.none(name="user_context"),# Vector de Memorias
            ],
            properties=common_props + [
                wc.Property(name="transcript_text", data_type=wc.DataType.TEXT),
                wc.Property(name="duration", data_type=wc.DataType.NUMBER),
            ],
        ),
        dict(
            name="Audio",
            description="Audio con multi-vector (Acústico, Semántico, Usuario)",
            vectorizer_config=[
                wc.Configure.NamedVectors.none(name="audio"),       # Wav2Vec (Acústico)
                wc.Configure.NamedVectors.none(name="transcript"),  # Texto (Letra/Voz)
                wc.Configure.NamedVectors.none(name="user_context"),# Vector de Memorias
            ],
            properties=common_props + [
                wc.Property(name="transcript_text", data_type=wc.DataType.TEXT),
                wc.Property(name="artist", data_type=wc.DataType.TEXT),
            ],
        ),
        dict(
            name="Document",
            description="Documentos y Textos con vector de contenido y contexto",
            vectorizer_config=[
                wc.Configure.NamedVectors.none(name="content"),      # Vector del contenido principal
                wc.Configure.NamedVectors.none(name="user_context"), # Vector de Memorias
            ],
            properties=common_props + [
                wc.Property(name="content", data_type=wc.DataType.TEXT),
                wc.Property(name="doc_type", data_type=wc.DataType.TEXT),
                wc.Property(name="author", data_type=wc.DataType.TEXT),
            ],
        ),
    ]

    existing = set(client.collections.list_all().keys())

    for cfg in collections_cfg:
        if cfg["name"] in existing:
            print(f'✔ Colección Weaviate "{cfg["name"]}" ya existe.')
            continue

        client.collections.create(
            name=cfg["name"],
            description=cfg["description"],
            properties=cfg["properties"],
            vectorizer_config=cfg["vectorizer_config"],
        )
        print(f'✅ Colección Weaviate "{cfg["name"]}" creada con Multi-Vectores.')

    client.close()