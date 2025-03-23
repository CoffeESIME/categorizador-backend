import json
from django.conf import settings
from neo4j import GraphDatabase
from api.embeddings_to_neo import store_embedding  # Usa tu store_embedding actual
from api.neo4j_client import driver                # Driver de Neo4j
from langchain_ollama import OllamaEmbeddings     # Para obtener embeddings con tu modelo local

def process_authors_data(authors_file="data.json", output_file="quotes_final.json"):
    """
    Lee un archivo JSON con la estructura:
    [
      {
        "author_id": "...",
        "name": "...",
        "birth_year": 1234,
        "death_year": 1235,
        "major_work": "...",
        "quotes": [
          {
            "quote_id": "...",
            "text": "...",
            "source": "...",
            "tags": [...]
          },
          ...
        ]
      },
      ...
    ]
    
    Para cada quote:
      - Calcula embeddings con OllamaEmbeddings
      - Usa store_embedding(...) para crear un nodo en Neo4j con la etiqueta 'Quote'
        (propiedades: doc_id=quote_id, embedding, meta)
      - Luego, crea nodos `Tag` y la relación `(:Quote)-[:HAS_TAG]->(:Tag)`.
      - Genera un diccionario final con la data de la quote { text, bibliography, tags }.

    Al final, escribe en `quotes_final.json` un objeto donde cada key es quote_id.
    """

    print("Iniciando proceso de autores...")

    # 1) Cargar la data de autores
    with open(authors_file, "r", encoding="utf-8") as f:
        authors_data = json.load(f)

    # 2) Inicializar el modelo de embeddings local de Ollama
    embedding_model = OllamaEmbeddings(model="granite-embedding:latest")

    # 3) Diccionario para guardar la info final (quotes)
    final_quotes = {}

    # 4) Recorrer autores y quotes
    for author in authors_data:
        author_name = author.get("name", "")
        major_work = author.get("major_work", "")

        for quote in author.get("quotes", []):
            quote_id = quote.get("quote_id")
            text = quote.get("text", "")
            source = quote.get("source", "")
            tags = quote.get("tags", [])

            # Generar 'bibliography' combinando la obra principal y el source
            bibliography = f"{major_work} - {source}"

            # 5) Calcular embeddings del texto de la quote
            embeddings = []
            if text:
                embeddings = embedding_model.embed_documents([text])

            # 6) Armar meta para store_embedding
            meta = {
                "author": author_name,
                "bibliography": bibliography,
                "text": text  # puedes omitir si no quieres guardar texto completo en Neo4j
            }

            # 7) Insertar en Neo4j usando store_embedding
            #    doc_id = quote_id, label = "Quote"
            node = store_embedding(
                doc_id=quote_id,
                embedding=embeddings,
                meta=meta,
                label="Quote"
            )
            print(f"Inserted {quote_id} as Quote node. Neo4j node info:", node)

            # 8) Crear nodos Tag y relaciones. 
            #    store_embedding crea un nodo (n:Quote). Ahora, con `driver.session()` relacionamos tags.
            if tags:
                with driver.session() as session:
                    for tag in tags:
                        tag = tag.strip()
                        if not tag:
                            continue
                        session.run("""
                            MERGE (t:Tag {name: $tag})
                            WITH t
                            MATCH (q:Quote {doc_id: $doc_id})
                            MERGE (q)-[:HAS_TAG]->(t)
                        """, tag=tag, doc_id=quote_id)

            # 9) Guardar en final_quotes para JSON final
            final_quotes[quote_id] = {
                "text": text,
                "bibliography": bibliography,
                "tags": tags
            }

    # 10) Escribir final_quotes en un archivo JSON
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(final_quotes, out, ensure_ascii=False, indent=4)

    print(f"Proceso completado. JSON final guardado en '{output_file}'.")


def main():
    process_authors_data(
        authors_file="data.json",
        output_file="quotes_final.json"
    )

if __name__ == "__main__":
    main()

