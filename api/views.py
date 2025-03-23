from .embeddings_to_neo import store_embedding
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import UploadedFileSerializer
import os
import json
from django.conf import settings
from rest_framework import status
from .models import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import uuid
from .neo4j_client import driver

class MultiFileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist("files")
        if not files:
            return Response({"error": "No se enviaron archivos"}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_files = []
        for file in files:
            instance = UploadedFile.objects.create(
                file=file,
                file_type=file.content_type,
                size=file.size,
                status='pending'
            )
            uploaded_files.append({
                "id": instance.id,
                "original_name": instance.original_name,
                "location": instance.file.url,
                "status": 'uploaded',
                "file_type": file.content_type
            })
        return Response({
            "status": "Archivos subidos correctamente",
            "files": uploaded_files
        }, status=status.HTTP_200_OK)

class PendingFilesView(APIView):
    def get(self, request, *args, **kwargs):
        pending_files = UploadedFile.objects.filter(status='pending')
        serializer = UploadedFileSerializer(pending_files, many=True)
        return Response({"files": serializer.data}, status=status.HTTP_200_OK)
class MetadataProcessingView(APIView):
    """
    Recibe una lista de objetos 'FileMetadata' y los procesa.
    Campos importantes en el JSON de entrada:
      - id: string (identificador del archivo, coincide con 'UploadedFile.file.name' o PK)
      - content, analysis, description ...
      - multilingual: boolean (si true, usar content_es, content_en, etc.)
      - chunks / vectorOfVectors: boolean (si true, hacer chunking)
      - deletedFile: boolean (si true, eliminar el archivo tras procesar)
      - ... (todos los campos que quieras)
    """

    def post(self, request):
        try:
            file_metadata_list = request.data
            if not isinstance(file_metadata_list, list):
                return Response({"error": "La entrada debe ser un array de objetos."},
                                status=status.HTTP_400_BAD_REQUEST)
            embedding_model = OllamaEmbeddings(model="granite-embedding:latest")
            results = []
            for meta in file_metadata_list:
                file_id = meta.get("id") 
                if not file_id:
                    results.append({"error": "No se recibió 'id' en los metadatos"})
                    continue
                try:
                    uploaded_file = UploadedFile.objects.get(original_name=file_id)
                except UploadedFile.DoesNotExist:
                    results.append({
                        "id": file_id,
                        "error": "El archivo no existe en la base de datos"
                    })
                    continue
                texts_for_embedding = []
                if meta.get("multilingual"):
                    for lang_code in meta.get("languages", []):
                        key = f"content_{lang_code}"
                        if key in meta and meta[key]:
                            texts_for_embedding.append(meta[key])
                else:
                    if meta.get("content"):
                        texts_for_embedding.append(meta["content"])
                    if meta.get("analysis"):
                        texts_for_embedding.append(meta["analysis"])
                    if meta.get("description"):
                        texts_for_embedding.append(meta["description"])

                do_chunking = meta.get("chunks") or meta.get("vectorOfVectors") or False
                chunked_texts = []
                if do_chunking:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    for txt in texts_for_embedding:
                        sub_docs = splitter.split_text(txt)
                        chunked_texts.extend(sub_docs)
                else:
                    chunked_texts = texts_for_embedding
                embeddings = []
            
                if embedding_model:
                    embeddings = embedding_model.embed_documents(chunked_texts)
                    doc_id = meta.get("id")
                    node = store_embedding(
                        doc_id=doc_id,
                        embedding=embeddings,
                        label="UnconnectedDoc",
                        meta=meta
                    )
                    print(node)
                json_filename = f"metadata_{uploaded_file.original_name}.json"
                json_path = os.path.join(settings.MEDIA_ROOT, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                if meta.get("deletedFile") is True:
                    file_path = uploaded_file.file.path
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    uploaded_file.status = "deleted"
                    uploaded_file.save()
                result_item = {
                    "id": file_id,
                    "filename": uploaded_file.file.name,
                    "status": uploaded_file.status,
                    "num_chunks": len(chunked_texts),
                    "embeddings_created": len(embeddings) if embeddings else 0
                }
                results.append(result_item)

            return Response({
                "status": "OK",
                "results": results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class UnconnectedNodesView(APIView):
    """
    Retorna nodos que no tienen ninguna relación (no conectados),
    mostrando las propiedades guardadas: doc_id, author, title, work, languages, sentiment_word,
    categories, keywords, content_type, tags, topics y style.
    """
    def get(self, request, *args, **kwargs):
        query = """
        MATCH (n:UnconnectedDoc)
        WHERE NOT (n)--()
        RETURN n.doc_id AS docId,
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
               labels(n) AS labels
        """
        results = []
        with driver.session() as session:
            records = session.run(query)
            for record in records:
                results.append({
                    "doc_id": record.get("docId"),
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
                    "labels": record.get("labels")
                })
        return Response(results, status=status.HTTP_200_OK)


class GraphView(APIView):
    """
    Endpoint para obtener un subgrafo filtrado.
    Parámetros opcionales (query parameters):
      - startingNodeId: id del nodo de inicio.
      - maxDepth: profundidad de la búsqueda (por defecto 2).
      - relationshipType: tipo de relación para filtrar.
      - nodeLabels: etiquetas (separadas por coma) para filtrar nodos.
      - limit: número máximo de nodos a retornar (por defecto 100).
    
    Retorna un objeto JSON con:
      { "nodes": [...], "edges": [...] }
    """
    def get(self, request, *args, **kwargs):
        starting_node_id = request.query_params.get("startingNodeId", None)
        try:
            max_depth = int(request.query_params.get("maxDepth", 2))
        except ValueError:
            max_depth = 2
        relationship_type = request.query_params.get("relationshipType", None)
        node_labels_str = request.query_params.get("nodeLabels", "")
        node_labels = (
            [label.strip() for label in node_labels_str.split(",") if label.strip()]
            if node_labels_str
            else []
        )
        try:
            limit = int(request.query_params.get("limit", 100))
        except ValueError:
            limit = 100

        nodes = []
        edges = []

        with driver.session() as session:
            if starting_node_id:
                # Buscar subgrafo a partir de un nodo de inicio, excluyendo nodos de tipo NodeType
                query_nodes = f"""
                    MATCH (start)
                    WHERE (start.doc_id = $startingNodeId OR start.id = $startingNodeId)
                      AND NOT start:NodeType
                    WITH start
                    MATCH (start)-[r*0..$maxDepth]-(n)
                    WHERE NOT n:NodeType
                """
                if node_labels:
                    query_nodes += " AND ANY(label IN labels(n) WHERE label IN $nodeLabels) "
                query_nodes += " RETURN DISTINCT n, labels(n) as labels"
                result = session.run(
                    query_nodes,
                    startingNodeId=starting_node_id,
                    maxDepth=max_depth,
                    nodeLabels=node_labels,
                )
            else:
                # Sin nodo de inicio: buscar nodos, excluyendo NodeType, con límite
                query_nodes = "MATCH (n) WHERE NOT n:NodeType "
                if node_labels:
                    query_nodes += " AND ANY(label IN labels(n) WHERE label IN $nodeLabels) "
                query_nodes += " RETURN n, labels(n) as labels LIMIT $limit"
                result = session.run(query_nodes, nodeLabels=node_labels, limit=limit)

            for record in result:
                node = dict(record.get("n"))
                # Usamos 'doc_id' si existe, o 'id' como respaldo para identificar el nodo
                node_id = node.get("doc_id") or node.get("id")
                node["nodeId"] = node_id
                node["labels"] = record.get("labels")
                nodes.append(node)

            # Obtener las relaciones (edges) entre los nodos recuperados
            node_ids = [node["nodeId"] for node in nodes if node.get("nodeId")]
            if node_ids:
                query_edges = """
                    MATCH (a)-[r]->(b)
                    WHERE (a.doc_id IN $nodeIds OR a.id IN $nodeIds)
                      AND (b.doc_id IN $nodeIds OR b.id IN $nodeIds)
                """
                if relationship_type:
                    query_edges += " AND type(r) = $relationshipType "
                query_edges += " RETURN a, b, type(r) as relation"
                
                params = {"nodeIds": node_ids}
                if relationship_type:
                    params["relationshipType"] = relationship_type
                
                edge_result = session.run(query_edges, **params)
                for record in edge_result:
                    a = dict(record.get("a"))
                    b = dict(record.get("b"))
                    source = a.get("doc_id") or a.get("id")
                    target = b.get("doc_id") or b.get("id")
                    edges.append({
                        "source": source,
                        "target": target,
                        "relation": record.get("relation")
                    })

        return Response({"nodes": nodes, "edges": edges}, status=status.HTTP_200_OK)


class NodeConnectionsView(APIView):
    """
    Endpoint para obtener las conexiones de un nodo específico.
    Se espera recibir el doc_id del nodo en la URL.
    Las conexiones se devuelven como nodos destino con las mismas propiedades de DocumentNode.
    """
    def get(self, request, nodeId, *args, **kwargs):
        query = """
        MATCH (n {doc_id: $nodeId})-[r]->(m)
        RETURN m.doc_id AS docId,
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
               labels(m) AS labels
        """
        connections = []
        with driver.session() as session:
            result = session.run(query, nodeId=nodeId)
            for record in result:
                connections.append({
                    "doc_id": record.get("docId"),
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
            "tag", "quote", "music", "language", "sentiment"
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
        MATCH (n:Document {{doc_id: $doc_id}})
        SET {set_clause}
        RETURN n
        """
        params = {"doc_id": doc_id}
        params.update(updates)
        with driver.session() as session:
            session.run(query, **params)
        return Response({"message": "Nodo actualizado", "doc_id": doc_id}, status=status.HTTP_200_OK)
class DeleteNodeConnectionView(APIView):
    """
    Endpoint para eliminar una conexión entre dos nodos.
    Recibe una petición DELETE a /api/nodes/<nodeId>/connections/<connectionNodeId>
    y elimina la relación CONNECTED_TO entre el nodo con doc_id=nodeId y el nodo con doc_id=connectionNodeId.
    """
    def delete(self, request, nodeId, connectionNodeId, *args, **kwargs):
        query = """
        MATCH (n {doc_id: $nodeId})-[r:CONNECTED_TO]->(m {doc_id: $connectionNodeId})
        DELETE r
        """
        try:
            with driver.session() as session:
                session.run(query, nodeId=nodeId, connectionNodeId=connectionNodeId)
            return Response(
                {"message": "Conexión eliminada exitosamente."},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )