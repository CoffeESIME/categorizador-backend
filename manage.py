#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import json
import uuid
from neo4j import GraphDatabase
from weaviate.connect import ConnectionParams
import weaviate 

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "s3cr3t2012"))

def seed_node_types():
    """
    Crea o actualiza (mediante MERGE) los nodos con etiqueta :NodeType 
    que definen los tipos de nodo base para la aplicación.
    
    Cada objeto del array 'node_types' representa la definición de un tipo de nodo, con:
      - id: Identificador interno del tipo
      - name: Nombre descriptivo
      - fields: Lista (serializada en JSON) de campos relevantes (fieldName, placeholder, required, etc.)
    """
    
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
                {"fieldName": "allowEmbedding", "placeholder": "¿Permitir embeddings? (true/false)", "required": False},
            ])
        },
        {
            "id": "music",
            "name": "Music",
            "fields": json.dumps([
                {"fieldName": "title", "placeholder": "Título de la canción/pieza", "required": True},
                {"fieldName": "doc_id", "placeholder": "Nombre de archivo o ID en el sistema", "required": True},
                {"fieldName": "artist", "placeholder": "Artista o banda", "required": False},
                {"fieldName": "on_pc", "placeholder": "¿Está almacenada localmente? (true/false)", "required": False},
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
    }
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
    """
    Asegura que existan los directorios necesarios para guardar los archivos
    según su tipo.
    """
    base_dir = os.path.join('uploads')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    subdirs = ['images', 'videos', 'audio', 'documents', 'texts', 'others']
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    
    print("✅ Directorios de uploads creados correctamente.")

def seed_weaviate_schema():
    """
    Conecta a Weaviate y crea las clases (por ejemplo, Imagenes, Textos, Audio, Video)
    si no existen.
    """
    
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

    classesVec = [
        {
            "class": "Imagenes",
            "description": "Clase para almacenar vectores de imágenes",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "file_location", "dataType": ["text"]},
                {"name": "analysis", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]}
            ]
        },
        {
            "class": "Textos",
            "description": "Clase para almacenar vectores de textos",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "author", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "analysis", "dataType": ["text"]},
                {"name": "file_location", "dataType": ["text"]},
                {"name": "doc_id", "dataType": ["text"]}
            ]
        },
        {
            "class": "Audio",
            "description": "Clase para almacenar vectores de audio",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "file_location", "dataType": ["text"]},
                {"name": "analysis", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]}
            ]
        },
        {
            "class": "Video",
            "description": "Clase para almacenar vectores de video",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "file_location", "dataType": ["text"]},
                {"name": "analysis", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]}
            ]
        }
    ]

    # Obtener todas las colecciones existentes
    try:
        existing_collections = client.collections.list_all()
        existing_names = set(existing_collections.keys())
        
        for cls in classesVec:
            class_name = cls["class"]
            if class_name in existing_names:
                print(f'La colección "{class_name}" ya existe en Weaviate.')
            else:
                # Crear la colección si no existe
                properties = []
                for prop in cls["properties"]:
                    properties.append({
                        "name": prop["name"],
                        "dataType": prop["dataType"][0]
                    })
                
                # Crear la colección
                collection = client.collections.create(
                    name=class_name,
                    description=cls["description"],
                    properties=properties
                )
                print(f'La colección "{class_name}" ha sido creada en Weaviate.')
    except Exception as e:
        print(f"Error al configurar el esquema de Weaviate: {str(e)}")
    
    client.close()

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'categorizador.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    ensure_upload_directories()
    seed_node_types()
    seed_weaviate_schema()
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()