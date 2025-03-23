import json
import uuid
from neo4j import GraphDatabase

# Configura la conexión a Neo4j según tu entorno
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

if __name__ == "__main__":
    seed_node_types()
