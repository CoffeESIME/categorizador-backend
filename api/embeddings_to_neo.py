# embeddings_to_neo4j.py
from .neo4j_client import driver

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
        "style"
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
    params = {"doc_id": doc_id}
    
    # Tratar el embedding por separado (asegurarse de que es una lista plana)
    if isinstance(embedding, list):
        if any(isinstance(elem, list) for elem in embedding):
            # Si embedding contiene listas, aplanarlo
            print("WARNING: Embedding contains nested lists, flattening...")
            params["embedding"] = flatten_list_recursive(embedding)
        else:
            params["embedding"] = embedding
    
    # Procesar el resto de propiedades una por una
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
            return record["n"] if record else None
    except Exception as e:
        print(f"ERROR with Neo4j: {str(e)}")
        print(f"Parameters causing issues: {params}")
        # Intentar identificar cuál parámetro está causando el problema
        for key, value in params.items():
            print(f"Testing {key}...")
            test_params = {"test_value": value}
            try:
                session.run("RETURN $test_value", test_params)
                print(f"  {key} is OK")
            except Exception as param_error:
                print(f"  ERROR with {key}: {str(param_error)}")
        raise  # Re-lanzar la excepción original