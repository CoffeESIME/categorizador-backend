import os
import json

from api.neo4j_client import driver
import weaviate
import weaviate.classes.config as wc
from django.conf import settings


def seed_node_types():
    """Crea o actualiza nodos base en Neo4j para la aplicación."""
    node_types = [
        {
            "id": "author",
            "name": "Author",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del autor", "required": True},
                {"fieldName": "birthdate", "placeholder": "Fecha de nacimiento (YYYY-MM-DD)", "required": False},
                {"fieldName": "bio", "placeholder": "Breve biografía", "required": False},
            ])
        },
        {
            "id": "image",
            "name": "Image",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título de la imagen", "required": True},
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True}
            ])
        },
        {
            "id": "video",
            "name": "Video",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título del video", "required": True},
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True},
                {"fieldName": "duration", "placeholder": "Duración en segundos", "required": False}
            ])
        },
        {
            "id": "book",
            "name": "Book",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título del libro", "required": True},
                {"fieldName": "authorName", "placeholder": "Autor del libro", "required": False},
                {"fieldName": "publication_year", "placeholder": "Año de publicación", "required": False},
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True},
            ])
        },
        {
            "id": "country",
            "name": "Country",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del país", "required": True},
                {"fieldName": "iso_code", "placeholder": "Código ISO del país (ej. MX, US)", "required": False},
            ])
        },
        {
            "id": "tag",
            "name": "Tag",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre de la etiqueta", "required": True},
                {"fieldName": "description", "placeholder": "Descripción breve", "required": False},
            ])
        },
        {
            "id": "quote",
            "name": "Quote",
            "fields": json.dumps([
                {"fieldName": "text", "placeholder": "Texto de la cita", "required": True},
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True},
                {"fieldName": "allowEmbedding", "placeholder": "¿Permitir embeddings? (True/False)", "required": False},
            ])
        },
        {
            "id": "music",
            "name": "Music",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título de la canción/pieza", "required": True},
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True},
                {"fieldName": "artist", "placeholder": "Artista o banda", "required": False},
                {"fieldName": "on_pc", "placeholder": "¿Está almacenada localmente? (True/False)", "required": False},
            ])
        },
        {
            "id": "language",
            "name": "Language",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del idioma (ej. Español, Inglés)", "required": True},
                {"fieldName": "iso_code", "placeholder": "Código del idioma (ej. es, en)", "required": False},
            ])
        },
        {
            "id": "UnconnectedDoc",
            "name": "Unconnected Document",
            "fields": json.dumps([
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True},
                {"fieldName": "author", "placeholder": "Autor del documento (opcional)", "required": False},
                {"fieldName": "title", "placeholder": "Título del documento (opcional)", "required": False},
                {"fieldName": "work", "placeholder": "Trabajo o proyecto relacionado (opcional)", "required": False},
                {"fieldName": "languages", "placeholder": "Lista de idiomas (ej. ['es','en'])", "required": False},
                {"fieldName": "sentiment_word", "placeholder": "Palabra de sentimiento (opcional)", "required": False},
                {"fieldName": "categories", "placeholder": "Categorías (ej. ['news','blog'])", "required": False},
                {"fieldName": "keywords", "placeholder": "Palabras clave (opcional)", "required": False},
                {"fieldName": "content_type", "placeholder": "Tipo de contenido (ej. image, video, etc.)", "required": False},
                {"fieldName": "tags", "placeholder": "Etiquetas (opcional)", "required": False},
                {"fieldName": "topics", "placeholder": "Temas (opcional)", "required": False},
                {"fieldName": "style", "placeholder": "Estilo (opcional)", "required": False},
                {"fieldName": "file_location", "placeholder": "Ubicación del archivo en el sistema", "required": False}
            ])
        },
        {
            "id": "post",
            "name": "Post",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título del post (opcional)", "required": False},
                {"fieldName": "post_id", "placeholder": "ID del post o URL única", "required": True},
                {"fieldName": "post_type", "placeholder": "Tipo (ej. Texto, Imagen, Cita, Enlace, Video)", "required": True},
                {"fieldName": "content", "placeholder": "Contenido principal o cuerpo del texto", "required": False},
                {"fieldName": "image_url", "placeholder": "URL de la imagen (si aplica)", "required": False},
                {"fieldName": "source_url", "placeholder": "URL de la fuente original (si es un reblog)", "required": False},
                {"fieldName": "source_author", "placeholder": "Autor de la fuente original (si se conoce)", "required": False},
                {"fieldName": "tags_string", "placeholder": "Etiquetas separadas por comas (ej. arte, filosofía)", "required": False},
                {"fieldName": "notes", "placeholder": "Tus notas o comentarios personales sobre el post", "required": False}
            ])
        },
        {
            "id": "concept",
            "name": "Concept",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del Concepto", "required": True},
                {"fieldName": "description", "placeholder": "Definición o resumen del concepto", "required": True},
                {"fieldName": "domain", "placeholder": "Dominio o campo (ej. Filosofía, Física)", "required": False}
            ])
        },
        {
            "id": "project",
            "name": "Project",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del Proyecto", "required": True},
                {"fieldName": "description", "placeholder": "Objetivo o resumen del proyecto", "required": False},
                {"fieldName": "status", "placeholder": "Estado (ej. Activo, Pausado, Completado)", "required": False},
                {"fieldName": "start_date", "placeholder": "Fecha de inicio (YYYY-MM-DD)", "required": False},
                {"fieldName": "end_date", "placeholder": "Fecha de finalización (YYYY-MM-DD)", "required": False}
            ])
        },
        {
            "id": "event",
            "name": "Event",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre del Evento", "required": True},
                {"fieldName": "date", "placeholder": "Fecha del evento (YYYY-MM-DD)", "required": False},
                {"fieldName": "location", "placeholder": "Lugar (ej. Ciudad, País, Online)", "required": False},
                {"fieldName": "description", "placeholder": "Descripción del evento", "required": False}
            ])
        },
        {
            "id": "organization",
            "name": "Organization",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre de la Organización", "required": True},
                {"fieldName": "type", "placeholder": "Tipo (ej. Universidad, Editorial, Empresa)", "required": False},
                {"fieldName": "website", "placeholder": "Sitio web oficial", "required": False}
            ])
        },
        {
            "id": "source",
            "name": "Source",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título de la fuente", "required": True},
                {"fieldName": "url", "placeholder": "URL del recurso en línea", "required": False},
                {"fieldName": "type", "placeholder": "Tipo de fuente (ej. Artículo web, Paper)", "required": False},
                {"fieldName": "access_date", "placeholder": "Fecha de acceso (YYYY-MM-DD)", "required": False},
                {"fieldName": "doc_id", "placeholder": "ID del archivo si aplica", "required": False}
            ])
        },
        {
            "id": "person",
            "name": "Person",
            "fields": json.dumps([
                {"fieldName": "name", "placeholder": "Nombre de la persona", "required": True},
                {"fieldName": "birthdate", "placeholder": "Fecha de nacimiento (YYYY-MM-DD)", "required": False},
                {"fieldName": "bio", "placeholder": "Breve biografía", "required": False},
                {"fieldName": "roles", "placeholder": "Roles (ej. Autor, Actor, Traductor)", "required": False}
            ])
        },
    ]

    with driver.session() as session:
        for node_type in node_types:
            query = """
            MERGE (nt:NodeType {id: $id})
            SET nt.name = $name,
                nt.fields = $fields
            """
            session.run(
                query,
                id=node_type["id"],
                name=node_type["name"],
                fields=node_type["fields"]
            )

    print("Tipos de nodo sembrados exitosamente.")


def ensure_upload_directories():
    """Crea los directorios requeridos para almacenar archivos subidos."""
    base_dir = os.path.join('uploads')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    subdirs = ['images', 'videos', 'audio', 'documents', 'texts', 'others']
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    print("✅ Directorios de uploads creados correctamente.")


def seed_weaviate_schema() -> None:
    """Crea colecciones en Weaviate si no existen."""
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

    collections_cfg = [
        dict(
            name="Imagenes",
            description="Imágenes con vectores CLIP, OCR y descripción",
            vectorizer_config=[
                wc.Configure.NamedVectors.none(name="vector_clip"),
                wc.Configure.NamedVectors.none(name="vector_ocr"),
                wc.Configure.NamedVectors.none(name="vector_des"),
            ],
            properties=[
                wc.Property(name="title", data_type=wc.DataType.TEXT),
                wc.Property(name="doc_id", data_type=wc.DataType.TEXT),
                wc.Property(name="file_location", data_type=wc.DataType.TEXT),
                wc.Property(name="analysis", data_type=wc.DataType.TEXT),
                wc.Property(name="content", data_type=wc.DataType.TEXT),
            ],
        ),
        dict(
            name="Textos",
            description="Documentos de texto (vector BYO)",
            vectorizer_config=wc.Configure.Vectorizer.none(),
            properties=[
                wc.Property(name="title", data_type=wc.DataType.TEXT),
                wc.Property(name="author", data_type=wc.DataType.TEXT),
                wc.Property(name="content", data_type=wc.DataType.TEXT),
                wc.Property(name="analysis", data_type=wc.DataType.TEXT),
                wc.Property(name="file_location", data_type=wc.DataType.TEXT),
                wc.Property(name="doc_id", data_type=wc.DataType.TEXT),
            ],
        ),
        dict(
            name="Audio",
            description="Archivos de audio",
                    vectorizer_config=[
            wc.Configure.NamedVectors.none(name="vector_audio"),
            wc.Configure.NamedVectors.none(name="vector_text"),
        ],
            properties=[
                wc.Property(name="title", data_type=wc.DataType.TEXT),
                wc.Property(name="doc_id", data_type=wc.DataType.TEXT),
                wc.Property(name="file_location", data_type=wc.DataType.TEXT),
                wc.Property(name="analysis", data_type=wc.DataType.TEXT),
                wc.Property(name="content", data_type=wc.DataType.TEXT),
            ],
        ),
        dict(
            name="Video",
            description="Archivos de video",
            vectorizer_config=[
            wc.Configure.NamedVectors.none(name="vector_video" ),
            wc.Configure.NamedVectors.none(name="vector_audio"),
            wc.Configure.NamedVectors.none(name="vector_text"  ),
            ],
            properties=[
                wc.Property(name="title", data_type=wc.DataType.TEXT),
                wc.Property(name="doc_id", data_type=wc.DataType.TEXT),
                wc.Property(name="file_location", data_type=wc.DataType.TEXT),
                wc.Property(name="analysis", data_type=wc.DataType.TEXT),
                wc.Property(name="content", data_type=wc.DataType.TEXT),
            ],
        ),
    ]

    existing = set(client.collections.list_all().keys())

    for cfg in collections_cfg:
        if cfg["name"] in existing:
            print(f'✔ "{cfg["name"]}" ya existe.')
            continue

        client.collections.create(
            name=cfg["name"],
            description=cfg["description"],
            properties=cfg["properties"],
            vectorizer_config=cfg["vectorizer_config"],
        )
        print(f'✅ Colección "{cfg["name"]}" creada.')

    client.close()

