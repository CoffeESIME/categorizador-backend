import re
import json
import uuid

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from langchain_ollama import OllamaEmbeddings
from api.neo4j_client import driver

from ..embeddings_to_neo import store_embedding
class UnconnectedNodesView(APIView):
    """
    Retorna nodos que no tienen ninguna relación (no conectados),
    mostrando las propiedades guardadas: doc_id, author, title, work, languages, sentiment_word,
    categories, keywords, content_type, tags, topics, style y file_location.
    """
    def get(self, request, *args, **kwargs):
        query = """
        MATCH (n:UnconnectedDoc)
        WHERE NOT (n)--()s
        RETURN n.id AS id,
               n.author AS author,
               n.title AS title,
               n.work AS work,
               n.languages AS languages,
               n.sentiment_word AS sentiment_word,
               n.categories AS categories,
               n.keywords AS keywords,
               n.content_type AS content_type,
               n.tags AS tags,
               n.topics AS topics,
               n.style AS style,
               n.doc_id AS doc_id,
               n.file_location AS file_location,
               labels(n) AS labels
        """
        results = []
        with driver.session() as session:
            records = session.run(query)
            for record in records:
                results.append({
    "doc_id": record.get("doc_id"),
    "author": record.get("author"),
    "title": record.get("title"),
    "work": record.get("work"),
    "languages": record.get("languages") if record.get("languages") is not None else [], # Devuelve [] si es None
    "sentiment_word": record.get("sentiment_word"),
    "categories": record.get("categories") if record.get("categories") is not None else [], # Devuelve [] si es None
    "keywords": record.get("keywords") if record.get("keywords") is not None else [], # Devuelve [] si es None
    "content_type": record.get("content_type"),
    "tags": record.get("tags") if record.get("tags") is not None else [], # Devuelve [] si es None
    "topics": record.get("topics") if record.get("topics") is not None else [], # Devuelve [] si es None
    "style": record.get("style"),
    "file_location": record.get("file_location"),
    "id": record.get("id"),
    "labels": record.get("labels")
})
        return Response(results, status=status.HTTP_200_OK)


class NodeConnectionsView(APIView):
    """
    Endpoint para obtener las conexiones de un nodo específico.
    Se espera recibir el doc_id del nodo en la URL.
    Las conexiones se devuelven como nodos destino con las mismas propiedades de DocumentNode.
    """
    def get(self, request, nodeId, *args, **kwargs): 
        
        query = """
        MATCH (n {id: $id})-[r]->(m)
        RETURN m.id AS id,
               m.author AS author,
               m.title AS title,
               m.work AS work,
               m.languages AS languages,
               m.sentiment_word AS sentiment_word,
               m.categories AS categories,
               m.keywords AS keywords,
               m.content_type AS content_type,
               m.tags AS tags,
               m.topics AS topics,
               m.style AS style,
               m.doc_id AS doc_id,
               m.file_location AS file_location,
               labels(m) AS labels
        """
        connections = []
        with driver.session() as session:
            # Corregido: parámetro 'id' en lugar de 'nodeId'
            result = session.run(query, id=nodeId)
            for record in result:
                connections.append({
                    "doc_id": record.get("doc_id"),
                    "author": record.get("author"),
                    "title": record.get("title"),
                    "work": record.get("work"),
                    "languages": record.get("languages"),
                    "sentiment_word": record.get("sentiment_word"),
                    "categories": record.get("categories"),
                    "keywords": record.get("keywords"),
                    "content_type": record.get("content_type"),
                    "tags": record.get("tags"),
                    "topics": record.get("topics"),
                    "style": record.get("style"),
                    "file_location": record.get("file_location"),
                    "id": record.get("id"),
                    "labels": record.get("labels")
                })
        return Response(connections, status=status.HTTP_200_OK)

class NodeTypesView(APIView):
    """
    GET: Retorna todos los tipos de nodo.
    POST: Crea un nuevo tipo de nodo.
    
    Se asume que los tipos se almacenan como nodos con label "NodeType"
    y tienen propiedades:
      - id: identificador único (string)
      - name: nombre del tipo
      - fields: JSON string con la definición de campos (por ejemplo, [{ "fieldName": "author", "required": true }, ...])
    """
    def get(self, request, *args, **kwargs):
        query = """
        MATCH (nt:NodeType)
        RETURN nt.id AS id, nt.name AS name, nt.fields AS fields
        """
        types = []
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                fields_str = record.get("fields")
                try:
                    fields = json.loads(fields_str) if fields_str else []
                except Exception:
                    fields = []
                types.append({
                    "id": record.get("id"),
                    "name": record.get("name"),
                    "fields": fields
                })
        return Response(types, status=status.HTTP_200_OK)
    
    def post(self, request, *args, **kwargs):
        # Payload esperado: { "name": "TipoEjemplo", "fields": [ { "fieldName": "author", "required": true }, ... ] }
        name = request.data.get("name")
        fields = request.data.get("fields", [])
        if not name:
            return Response({"error": "El campo 'name' es requerido."}, status=status.HTTP_400_BAD_REQUEST)
        # Generar un id único para el tipo
        type_id = str(uuid.uuid4())
        # Convertir fields a string JSON para almacenarlo en Neo4j
        fields_str = json.dumps(fields)
        query = """
        CREATE (nt:NodeType {id: $id, name: $name, fields: $fields})
        RETURN nt
        """
        with driver.session() as session:
            session.run(query, id=type_id, name=name, fields=fields_str)
        return Response({"id": type_id, "name": name, "fields": fields}, status=status.HTTP_201_CREATED)


class NodeCreationView(APIView):
    """
    Endpoint para crear nodos base con tipos predefinidos.
    Se espera un payload con:
      - type: Tipo de nodo (p.ej., "Autor", "Cita", "Obra", etc.)
      - properties: Objeto JSON con las propiedades del nodo (pueden ser propiedades básicas o extendidas).
    """

    ALLOWED_TYPES = [
    "author", "image", "video", "book", "country", 
    "tag", "quote", "music", "language", "sentiment"
]

    def post(self, request, *args, **kwargs):
        node_type = request.data.get("type")
        if not node_type or node_type not in self.ALLOWED_TYPES:
            return Response(
                {
                    "error": "El tipo de nodo es requerido y debe ser uno de los siguientes: " +
                             ", ".join(self.ALLOWED_TYPES)
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Obtiene las propiedades pasadas en la solicitud; si no hay, se inicia con un diccionario vacío.
        properties = request.data.get("properties", {})
        
        # Genera un ID único para el nodo y lo asigna a las propiedades
        node_id = str(uuid.uuid4())
        properties["id"] = node_id
        properties["type"] = node_type  # Puedes guardar el tipo como propiedad si te resulta útil

        # Construye la consulta Cypher: se usa interpolación de cadena para insertar la etiqueta (confiando en que el tipo es validado)
        query = f"CREATE (n:{node_type} $props) RETURN n"

        try:
            with driver.session() as session:
                result = session.run(query, props=properties)
                record = result.single()
                if record:
                    # Convertir el objeto neo4j.Node a un diccionario
                    node = record["n"]
                    return Response(dict(node), status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"error": "No se pudo crear el nodo."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

class NodesByTypeView(APIView):
    """
    Endpoint para obtener nodos filtrados por tipo.

    Se espera que 'nodeType' sea recibido como parámetro en la URL.
    Se utiliza la definición del nodo (almacenada en NodeType) para construir
    la proyección de las propiedades a retornar, ya que cada tipo tiene propiedades distintas.
    """
    def get(self, request, nodeType, *args, **kwargs):
        ALLOWED_TYPES = [
            "author", "image", "video", "book", "country", 
            "tag", "quote", "music", "language", "sentiment", 'UnconnectedDoc'
        ]
        if nodeType not in ALLOWED_TYPES:
            return Response(
                {"error": f"El tipo de nodo '{nodeType}' no es permitido."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        with driver.session() as session:
            # Primero, se obtiene la definición de campos para el tipo (almacenada en NodeType)
            type_query = """
            MATCH (nt:NodeType {id: $nodeType})
            RETURN nt.fields AS fields
            """
            type_result = session.run(type_query, nodeType=nodeType)
            record = type_result.single()
            if record and record.get("fields"):
                try:
                    fields_list = json.loads(record.get("fields"))
                except Exception as e:
                    fields_list = []
            else:
                fields_list = []
            
            # Se construye la lista de campos, iniciando siempre con "id"
            field_names = ["id"]
            for field in fields_list:
                if field.get("fieldName") and field.get("fieldName") not in field_names:
                    field_names.append(field.get("fieldName"))
            
            # Si no se obtuvieron campos, se usa una proyección por defecto (incluyendo "id")
            if not field_names:
                field_names = ["id", "author", "title", "work", "languages",
                               "sentiment_word", "categories", "keywords", "content_type",
                               "tags", "topics", "style"]

            # Construir la cadena de proyección: para cada campo, usar un alias si es necesario
            projection_parts = []
            for field in field_names:
                if field == "id":
                    projection_parts.append("n.id AS id")
                else:
                    projection_parts.append(f"n.{field} AS {field}")
            # Agregar las etiquetas del nodo
            projection_parts.append("labels(n) AS labels")
            projection = ",\n".join(projection_parts)
            
            query = f"""
            MATCH (n:{nodeType})
            RETURN {projection}
            """
            
            results = []
            records = session.run(query)
            for rec in records:
                node = {}
                for key in rec.keys():
                    node[key] = rec.get(key)
                results.append(node)
        
        return Response(results, status=status.HTTP_200_OK)
    

class UpdateNodeView(APIView):
    """
    Endpoint para actualizar un nodo existente.
    La URL debe incluir el id del nodo.
    Se espera recibir en el body los campos a actualizar, por ejemplo:
    { "name": "Nuevo Nombre", "properties": { "campo1": "nuevo valor", ... } }
    """
    def patch(self, request, id, *args, **kwargs):
        # Se pueden actualizar propiedades directamente; se espera un JSON con los campos a actualizar.
        updates = request.data
        if not updates:
            return Response({"error": "No se proporcionaron campos para actualizar."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Construir dinámicamente la cláusula SET
        set_clause = ", ".join([f"n.{key} = ${key}" for key in updates.keys()])
        query = f"""
        MATCH (n:Document {{id: $id}})
        SET {set_clause}
        RETURN n
        """
        params = {"id": id}
        params.update(updates)
        with driver.session() as session:
            session.run(query, **params)
        return Response({"message": "Nodo actualizado", "did": id}, status=status.HTTP_200_OK)

class DeleteNodeConnectionView(APIView):
    """
    Endpoint para eliminar una conexión entre dos nodos.
    Recibe una petición DELETE con:
    {
        "sourceId": "id_del_nodo_origen",
        "targetId": "id_del_nodo_destino",
        "relationshipType": "TIPO_DE_RELACION" (opcional, si no se especifica elimina cualquier relación)
    }
    """
    def delete(self, request, *args, **kwargs):
        source_node_id = request.data.get("sourceId")
        target_node_id = request.data.get("targetId")
        relationship_type_from_request = request.data.get("relationshipType")

        if not source_node_id or not target_node_id:
            return Response(
                {"error": "Se requieren los IDs de origen y destino."},
                status=status.HTTP_400_BAD_REQUEST
            )

        params = {
            "nodeId": source_node_id,
            "connectionNodeId": target_node_id
        }
        
        if relationship_type_from_request:

            if not re.match(r"^[a-zA-Z0-9_]+$", relationship_type_from_request):
                return Response(
                    {"error": "El tipo de relación contiene caracteres no válidos."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            query = """
            MATCH (n {id: $nodeId})-[r]->(m {id: $connectionNodeId})
            WHERE type(r) = $relType
            DELETE r
            """
            params["relType"] = relationship_type_from_request
        else:
            query = """
            MATCH (n {id: $nodeId})-[r]->(m {id: $connectionNodeId})
            DELETE r
            """
        try:
            with driver.session() as session:
                result = session.run(query, **params)
                summary = result.consume() # Obtener el resumen del resultado
                
                # Obtener el conteo de relaciones eliminadas del summary
                deleted_count = summary.counters.relationships_deleted

                if deleted_count > 0:
                    return Response(
                        {
                            "message": "Conexión(es) eliminada(s) exitosamente.",
                            "deletedCount": deleted_count
                        },
                        status=status.HTTP_200_OK
                    )
                else:
                    return Response(
                        {"error": "No se encontró o eliminó la conexión especificada (verifique IDs y tipo de relación)."},
                        status=status.HTTP_404_NOT_FOUND
                    )
        except Exception as e:
            return Response(
                {"error": f"Error interno del servidor: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



class ConnectNodesView(APIView):
    """
    Endpoint para conectar dos nodos con una relación.
    
    POST: Recibe una petición con:
    {
      "sourceNodeId": "id_del_nodo_origen",
      "targetNodeId": "id_del_nodo_destino",
      "relationshipType": "TIPO_DE_RELACION" (opcional, por defecto "CONNECTED_TO"),
      "relationshipProperties": {
        "propiedad1": valor1,
        "propiedad2": valor2,
        ...
      } (opcional)
    }
    
    Crea una relación entre los nodos correspondientes, con las propiedades especificadas.
    """
    def post(self, request, *args, **kwargs):
        source_node_id = request.data.get("sourceId")
        target_node_id = request.data.get("targetId")
        relationship_type = request.data.get("relationshipType", "CONNECTED_TO")
        relationship_properties = request.data.get("relationshipProperties", {})
        
        if not source_node_id or not target_node_id:
            return Response(
                {"error": "Se requieren los IDs de origen y destino."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validar que los nodos existen
        query_validate = """
        MATCH (source) 
        WHERE source.id = $sourceId
        WITH count(source) as sourceExists
        MATCH (target) 
        WHERE target.id = $targetId
        WITH sourceExists, count(target) as targetExists
        RETURN sourceExists > 0 AND targetExists > 0 as nodesExist
        """
        
        # Crear la relación con propiedades
        query_connect = """
        MATCH (source) 
        WHERE source.id = $sourceId
        MATCH (target) 
        WHERE target.id = $targetId
        MERGE (source)-[r:`{}`]->(target)
        SET r += $relProps
        RETURN source, target, r
        """.format(relationship_type)
        
        try:
            with driver.session() as session:
                # Primero validar que ambos nodos existen
                validate_result = session.run(query_validate, 
                                             sourceId=source_node_id, 
                                             targetId=target_node_id)
                record = validate_result.single()
                
                if not record or not record.get("nodesExist"):
                    return Response(
                        {"error": "Uno o ambos nodos no existen."},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Crear la relación con las propiedades
                result = session.run(query_connect, 
                                    sourceId=source_node_id, 
                                    targetId=target_node_id,
                                    relProps=relationship_properties)
                
                # Verificar el resultado
                if result.peek():
                    return Response(
                        {
                            "message": "Nodos conectados exitosamente.",
                            "sourceId": source_node_id,
                            "targetId": target_node_id,
                            "relationshipType": relationship_type,
                            "relationshipProperties": relationship_properties
                        },
                        status=status.HTTP_201_CREATED
                    )
                else:
                    return Response(
                        {"error": "No se pudo crear la conexión."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ConnectUnconnectedNodeView(APIView):
    """
    Endpoint para conectar automáticamente un nodo de tipo "UnconnectedDoc" 
    utilizando LLM local para generar las conexiones adecuadas.
    
    POST: Recibe una petición con:
    {
      "nodeId": "id_del_nodo_desconectado"
    }
    
    Utiliza LLM local para analizar propiedades y crear conexiones con nodos existentes.
    Si todas las propiedades se pueden transferir a nodos existentes, elimina el nodo original.
    """
    def post(self, request, *args, **kwargs):
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.llms import Ollama
        import json
        
        node_id = request.data.get("nodeId")
        
        if not node_id:
            return Response(
                {"error": "Se requiere el ID del nodo desconectado."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Verificar que el nodo existe y es de tipo UnconnectedDoc
        query_validate = """
        MATCH (n:UnconnectedDoc) 
        WHERE n.id = $nodeId
        RETURN n
        """
        
        # Obtener todos los tipos de nodos permitidos (excluyendo UnconnectedDoc)
        allowed_types_query = """
        MATCH (nt:NodeType)
        WHERE nt.id <> 'unconnected'
        RETURN nt.id AS id, nt.name as name
        """
        
        try:
            with driver.session() as session:
                # Verificar que el nodo existe y es del tipo correcto
                validate_result = session.run(query_validate, nodeId=node_id)
                record = validate_result.single()
                
                if not record:
                    return Response(
                        {"error": "El nodo no existe o no es de tipo UnconnectedDoc."},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Obtener las propiedades del nodo desconectado
                unconnected_node = dict(record.get("n"))
                
                # Obtener los tipos de nodos permitidos
                allowed_types = []
                allowed_types_result = session.run(allowed_types_query)
                for type_record in allowed_types_result:
                    allowed_types.append({
                        "id": type_record.get("id"),
                        "name": type_record.get("name")
                    })
                
                # Inicializar el modelo LLM
                llm = Ollama(
                    model=settings.DEFAULT_LLM_MODEL,
                    base_url=settings.LLM_BASE_URL,
                )
                
                # Construir el prompt para el LLM
                prompt = f"""
                Analiza este nodo desconectado y sus propiedades: {json.dumps(unconnected_node, indent=2)}
                
                Los tipos de nodos permitidos en el sistema son: {json.dumps(allowed_types, indent=2)}
                
                Genera una consulta Cypher para Neo4j que:
                1. Identifique nodos existentes para conectar con este UnconnectedDoc
                2. Cree las relaciones apropiadas basadas en las propiedades
                3. Si es posible transferir todas las propiedades relevantes, elimine el nodo original
                
                La consulta debe incluir búsquedas por similitud semántica y nombres/títulos exactos.
                Retorna SOLO la consulta Cypher, sin explicaciones adicionales.
                """
                
                # Ejecutar el LLM para generar la consulta
                cypher_query = llm.invoke(prompt).strip()
                
                # Ejecutar la consulta generada por el LLM
                try:
                    result = session.run(cypher_query)
                    summary = result.consume()
                    
                    # Verificar si el nodo aún existe (no fue eliminado por la consulta)
                    check_query = """
                    MATCH (n:UnconnectedDoc) 
                    WHERE n.id = $nodeId
                    RETURN n
                    """
                    check_result = session.run(check_query, nodeId=node_id)
                    node_still_exists = check_result.peek() is not None
                    
                    return Response({
                        "message": "Procesamiento del nodo completado.",
                        "nodeId": node_id,
                        "nodeEliminated": not node_still_exists,
                        "querySummary": {
                            "counters": {
                                "relationships_created": summary.counters.relationships_created,
                                "nodes_deleted": summary.counters.nodes_deleted
                            }
                        }
                    }, status=status.HTTP_200_OK)
                    
                except Exception as query_error:
                    return Response({
                        "error": f"Error al ejecutar la consulta generada: {str(query_error)}",
                        "generatedQuery": cypher_query
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json

def serialize_neo4j_value(value):
    """Serializa valores de Neo4j, especialmente tipos temporales."""
    if hasattr(value, 'isoformat'): # Para DateTime, Date, Time, etc.
        return value.isoformat()
    # Puedes añadir más conversiones aquí para tipos espaciales de Neo4j, etc.
    elif isinstance(value, list):
        return [serialize_neo4j_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_neo4j_value(v) for k, v in value.items()}
    return value

def serialize_neo4j_node_properties(node_data):
    if not node_data:
        return {}
    serialized_node = {}
    for key, value in node_data.items():
        serialized_node[key] = serialize_neo4j_value(value)
    return serialized_node


class IngestAuthorDataView(APIView):
    """
    Endpoint híbrido para la ingesta completa de un objeto de autor.
    1. Genera embeddings para cada cita y los guarda en Weaviate.
    2. Crea la estructura de nodos y relaciones en Neo4j en una transacción.
    El ``quote_id`` se usa como enlace entre ambos sistemas.
    """

    def post(self, request, *args, **kwargs):
        author_data = request.data

        if not all(k in author_data for k in ["author_id", "name", "quotes"]):
            return Response(
                {"error": "El JSON debe contener al menos 'author_id', 'name', y 'quotes'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Asegurar que exista la lista de temas para evitar errores en Cypher
        author_data.setdefault("temas", [])

        # --- 1. Inicialización de Clientes ---
        try:
            embedding_model = OllamaEmbeddings(
                model=settings.DEFAULT_EMBED_MODEL,
                base_url=settings.LLM_BASE_URL,
            )
            weaviate_client = CLIENT
        except Exception as e:
            return Response(
                {"error": f"Error al inicializar clientes: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # --- 2. Preparación de Datos para Weaviate (Batch) ---
        quotes_to_embed = author_data.get("quotes", [])
        if not quotes_to_embed:
            return Response({"message": "No hay citas para procesar."}, status=status.HTTP_200_OK)

        quote_texts = [q.get("text", "") for q in quotes_to_embed]
        quote_vectors = embedding_model.embed_documents(quote_texts)

        weaviate_objects_to_add = []
        for i, quote in enumerate(quotes_to_embed):
            properties = {
                "doc_id": quote.get("quote_id"),
                "author": author_data.get("name"),
                "title": f"Cita de {author_data.get('name')}",
                "content": quote.get("text"),
                "source": quote.get("source"),
            }
            weaviate_objects_to_add.append({"properties": properties, "vector": quote_vectors[i]})

        # --- 3. Ingesta por Lotes en Weaviate ---
        try:
            with weaviate_client.batch as batch:
                batch.batch_size = 100
                for obj in weaviate_objects_to_add:
                    batch.add_data_object(
                        data_object=obj["properties"],
                        class_name="Textos",
                        vector=obj["vector"],
                    )
        except Exception as e:
            return Response(
                {"error": f"Error durante la ingesta en Weaviate: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # --- 4. Ingesta Transaccional en Neo4j ---
        ingest_query = """
        // Recibe todo el objeto JSON como un parámetro $data
        WITH $data AS author

        // 1. Crea o encuentra al autor (Person) usando MERGE para evitar duplicados.
        MERGE (p:Person {author_id: author.author_id})
        ON CREATE SET
            p.name = author.name,
            p.birth_year = author.birth_year,
            p.death_year = author.death_year,
            p.major_work = author.major_work

        // 2. Procesa la lista de 'temas' del autor.
        // Cada 'tema' se convierte en un nodo 'Concept'.
        // La relación (Person)-[IS_ABOUT]->(Concept) representa los temas generales del autor.
        WITH p, author
        UNWIND author.temas AS tema_name
        MERGE (c:Concept {name: tema_name})
        MERGE (p)-[:IS_ABOUT]->(c)

        // 3. Procesa la lista de 'quotes' del autor.
        WITH DISTINCT p, author
        UNWIND author.quotes AS quote_data

        // Crea o encuentra la cita (Quote)
        MERGE (q:Quote {quote_id: quote_data.quote_id})
        ON CREATE SET
            q.text = quote_data.text,
            q.source = quote_data.source

        // Crea la relación (Person)-[AUTHORED_BY]->(Quote)
        MERGE (p)-[:AUTHORED_BY]->(q)

        // 4. Procesa los 'tags' de cada cita.
        // Interpretamos cada 'tag' de la cita como un 'Concept' sobre el cual trata la cita.
        // Esto es mucho más potente que usar la relación TAGGED_AS.
        WITH q, quote_data
        UNWIND quote_data.tags AS concept_name_from_tag
        MERGE (c_tag:Concept {name: concept_name_from_tag})

        // Crea la relación (Quote)-[IS_ABOUT]->(Concept)
        MERGE (q)-[:IS_ABOUT]->(c_tag)

        // 5. Devolvemos un resumen de lo que se creó o encontró.
        RETURN count(DISTINCT p) AS authors,
               count(DISTINCT q) AS quotes,
               count(DISTINCT c_tag) + count(DISTINCT c) AS concepts
        """

        try:
            with driver.session() as session:
                summary = session.execute_write(
                    lambda tx: tx.run(ingest_query, data=author_data).single()
                )

                response_data = {
                    "message": "Ingesta completada: Grafo creado en Neo4j y vectores guardados en Weaviate.",
                    "author_id": author_data.get("author_id"),
                    "neo4j_summary": dict(summary) if summary else {},
                    "weaviate_summary": {"vectors_added": len(weaviate_objects_to_add)},
                }
                return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {"error": f"Error en la transacción de Neo4j: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
