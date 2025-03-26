#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import json
import uuid
from neo4j import GraphDatabase
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
            # Usamos MERGE para que si el nodo ya existe por ID, solo se actualice
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
    # Crear directorio base si no existe
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Crear subdirectorios para cada tipo de archivo
    subdirs = ['images', 'videos', 'audio', 'documents', 'texts', 'others']
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    
    print("✅ Directorios de uploads creados correctamente.")


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
    
    # Crear directorios de uploads antes de ejecutar
    ensure_upload_directories()
    seed_node_types()
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
