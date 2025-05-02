from .embeddings_to_neo import limpiar_meta, store_embedding, guardar_imagen_en_weaviate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import UploadedFileSerializer
import os
import json
from django.conf import settings
from .models import UploadedFile
from langchain_ollama import OllamaEmbeddings
import uuid
from .neo4j_client import driver
import torch, clip
from PIL import Image
from .pdf_embeddings import embed_pdf_and_store
from .weaviate_client import CLIENT  
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip = None
preprocess = None
try:
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Modelo CLIP 'ViT-B/32' cargado exitosamente en el dispositivo: {device}")
except Exception as e:
    print(f"Error crítico al cargar el modelo CLIP: {e}")
    print("El procesamiento de imágenes estará deshabilitado.")


class MultiFileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist("files")
        if not files:
            return Response({"error": "No se enviaron archivos"}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_files = []
        for file in files:
            original_name = file.name
            sanitized_name = original_name.replace(" ", "_")
            file.name = sanitized_name
            
            instance = UploadedFile.objects.create(
                file=file,
                file_type=file.content_type,
                size=file.size,
                status='pending',
                original_name=sanitized_name  
            )
            uploaded_files.append({
                "id": instance.id,
                "original_name": sanitized_name,  
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
    Se admite un campo opcional "embedding_type" en cada objeto, que puede ser:
      - "image": Para obtener embedding de imagen usando CLIP.
      - "ocr": Para procesar texto extraído de imagen (OCR).
      - "text": Para embeddings de texto (contenido, análisis, descripción).
      - "audio": Para embeddings de audio.
      - "video": Para embeddings de video.
      - "graph": Para embeddings de grafos.
    Si no se especifica, se asume "text".
    
    Los embeddings se almacenarán en Weaviate según su tipo,
    mientras que los metadatos se guardarán en Neo4j como nodos UnconnectedDoc.
    """

    def post(self, request):
        try:
            file_metadata_list = request.data
            if not isinstance(file_metadata_list, list):
                return Response(
                    {"error": "La entrada debe ser un array de objetos."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            results = []
            for meta in file_metadata_list:
                file_id = meta.get("id")
                print('data', file_id)
                if not file_id:
                    results.append({"error": "No se recibió 'id' en los metadatos"})
                    continue
                try:
                    uploaded_file = UploadedFile.objects.get(original_name=file_id)
                    # Agregar la ubicación del archivo a los metadatos (si existe)
                    if uploaded_file.file_location:
                        meta["file_location"] = uploaded_file.file_location
                except UploadedFile.DoesNotExist:
                    results.append({
                        "id": file_id,
                        "error": "El archivo no existe en la base de datos"
                    })
                    continue
                embedding_type = meta.get("embedding_type", "text").lower()
                meta.setdefault("content_type", embedding_type)

                embedding = []
                uploads_dir = os.path.join('uploads', meta.get("file_location"))
                print("embedding", embedding_type)
                if embedding_type == "pdf":
                    pdf_path = os.path.join("uploads", meta["file_location"])
                    try:
                        embeddings_created, uuids = embed_pdf_and_store(
                            pdf_path=pdf_path,
                            original_doc_id=file_id,
                            client=CLIENT,
                        )
                        store_embedding(
                            doc_id=file_id,
                            embedding=[],          # vacío ➜ sólo metadatos
                            label="UnconnectedDoc",
                            meta=meta,
                        )
                    except Exception as pdf_err:
                        results.append({"id": file_id, "error": str(pdf_err)})
                        continue
                if embedding_type == "image_w_des":
                    image_path = uploads_dir
                    if not image_path or not os.path.exists(image_path):
                        results.append({
                            "id": file_id,
                            "error": "No se encontró la imagen en 'file_location'"
                        })
                        continue
                    try:
                        image = Image.open(image_path).convert("RGB") # Asegurar formato RGB
                        image_input = preprocess(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = model_clip.encode_image(image_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        embeddingClip = image_features.cpu().numpy()[0].tolist()
                        texts_for_embedding = []
                        embedding_des = []
                        propiedades = limpiar_meta(meta)
                        if meta.get("style"):
                            texts_for_embedding.append(meta["style"])
                        if meta.get("composition"):
                            texts_for_embedding.append(meta["composition"])
                        # if meta.get("color_palette"):
                        #     texts_for_embedding.append(meta["color_palette"])
                        if meta.get("analysis"):
                            texts_for_embedding.append(meta["analysis"])
                        if meta.get("description"):
                            texts_for_embedding.append(meta["description"])
                        combined_text = " ".join(texts_for_embedding)
                        if combined_text:
                            embedding_model = OllamaEmbeddings(model="granite-embedding:latest")
                            embedding_des = embedding_model.embed_documents([combined_text])[0]
                        uuid = guardar_imagen_en_weaviate(
                            client=CLIENT,
                            meta=propiedades,
                            vec_clip=embeddingClip,     # ← embedding visual
                            #vec_ocr=vector_ocr,       # ← embedding OCR (si lo tienes)
                            vec_desc=embedding_des,     # ← embedding descripción (si lo tienes)
                        )
                                                
                        
                        store_embedding(
                            doc_id=file_id,
                            embedding=[],
                            label="UnconnectedDoc", 
                            meta=meta,
                        )
                    except Exception as e:
                        print(f"  Error al procesar la imagen '{image_path}': {e}")
                        results.append({
                            "id": file_id,
                            "error": f"Error procesando imagen: {e}"
                        })
                        continue 

                elif embedding_type == "ocr_w_img":
                    ocr_text = meta.get("ocr_text", "")
                    if ocr_text:
                        meta["content"] = ocr_text  # Guardar el texto OCR como contenido
                        embedding_model = OllamaEmbeddings(model="granite-embedding:latest")
                        embedding_text = embedding_model.embed_documents([ocr_text])[0]
                        image = Image.open(image_path).convert("RGB") # Asegurar formato RGB
                        image_input = preprocess(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = model_clip.encode_image(image_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        embeddingClip = image_features.cpu().numpy()[0].tolist()
                        propiedades = limpiar_meta(meta)
                        uuid = guardar_imagen_en_weaviate(
                            client=CLIENT,
                            meta=propiedades,
                            vec_clip=embeddingClip,     # ← embedding visual
                            vec_ocr=embedding_text,       # ← embedding OCR (si lo tienes)
                        )
                        store_embedding(
                            doc_id=file_id,
                            embedding=[],
                            label="UnconnectedDoc",
                            meta=meta,
                        )
                    else:
                        results.append({
                            "id": file_id,
                            "error": "No se encontró 'ocr_text' para procesamiento OCR"
                        })
                        continue

                elif embedding_type == "text":
                    # Procesar textos: content, analysis, description, etc.
                    texts_for_embedding = []
                    if meta.get("ocr_text"):
                        texts_for_embedding.append(meta["ocr_text"])
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
                    combined_text = " ".join(texts_for_embedding)
                    if combined_text:
                        embedding_model = OllamaEmbeddings(model="granite-embedding:latest")
                        embedding = embedding_model.embed_documents([combined_text])[0]

                # elif embedding_type == "audio":
                #     # Ejemplo: usar un modelo de audio embeddings (reemplaza por tu implementación)
                #     from some_audio_embedding_module import AudioEmbeddings
                #     audio_path = meta.get("file_location")
                #     audio_model = AudioEmbeddings(model="your-audio-model")
                #     embedding = audio_model.embed(audio_path)

                # elif embedding_type == "video":
                #     # Ejemplo: extraer y combinar embeddings de frames clave de un video
                #     from some_video_embedding_module import VideoEmbeddings
                #     video_path = meta.get("file_location")
                #     video_model = VideoEmbeddings(model="your-video-model")
                #     embedding = video_model.embed(video_path)

                # elif embedding_type == "graph":
                #     # Ejemplo: procesar datos de grafo
                #     from some_graph_embedding_module import GraphEmbeddings
                #     graph_data = meta.get("graph_data")
                #     graph_model = GraphEmbeddings(model="your-graph-model")
                #     embedding = graph_model.embed(graph_data)

                else:
                    # Fallback: tratar el contenido como texto
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
                    combined_text = " ".join(texts_for_embedding)
                    if combined_text:
                        embedding_model = OllamaEmbeddings(model="granite-embedding:latest")
                        embedding = embedding_model.embed_documents([combined_text])[0]
                uploaded_file.status = "vectorized"
                uploaded_file.save()

                # Guardar metadatos en un archivo JSON (opcional)
                json_filename = f"metadata_{uploaded_file.original_name}.json"
                json_path = os.path.join(settings.MEDIA_ROOT, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # Eliminar el archivo si se indica
                if meta.get("deletedFile") is True:
                    file_path = uploaded_file.file.path
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    uploaded_file.status = "deleted"
                    uploaded_file.save()

                result_item = {
                    "id": file_id,
                    "filename": uploaded_file.file.name,
                    "file_location": uploaded_file.file_location,
                    "status": uploaded_file.status,
                    "embedding_type": embedding_type,
                    "embeddings_created": 1 if embedding else 0
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
    categories, keywords, content_type, tags, topics, style y file_location.
    """
    def get(self, request, *args, **kwargs):
        query = """
        MATCH (n:UnconnectedDoc)
        WHERE NOT (n)--()
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
                    WHERE start.id = $startingNodeId
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
        relationship_type = request.data.get("relationshipType")

        if not source_node_id or not target_node_id:
            return Response(
                {"error": "Se requieren los IDs de origen y destino."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Construir la consulta según si se especificó un tipo de relación
        if relationship_type:
            query = """
            MATCH (n {id: $nodeId})-[r:`{}`]->(m {id: $connectionNodeId})
            DELETE r
            RETURN count(r) as deletedRelations
            """.format(relationship_type)
        else:
            query = """
            MATCH (n {id: $nodeId})-[r]->(m {id: $connectionNodeId})
            DELETE r
            RETURN count(r) as deletedRelations
            """

        try:
            with driver.session() as session:
                result = session.run(query, 
                                   nodeId=source_node_id, 
                                   connectionNodeId=target_node_id)
                record = result.single()
                deleted_count = record["deletedRelations"] if record else 0

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
                        {"error": "No se encontró la conexión especificada."},
                        status=status.HTTP_404_NOT_FOUND
                    )
        except Exception as e:
            return Response(
                {"error": str(e)},
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
                llm = Ollama(model="mistral")
                
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

class TextProcessView(APIView):
    """
    Endpoint para procesar texto simple con un LLM.
    
    POST: Recibe un texto y lo procesa con un modelo de lenguaje para extraer metadatos.
    
    Parámetros esperados:
    {
      "input_text": "texto a procesar",
      "model": "nombre_del_modelo" (opcional, default: "deepseek-r1:32b"),
      "task": "text" (opcional),
      "temperature": valor_temperatura (opcional, default: 0.7)
    }
    
    Retorna: metadatos estructurados con author, title, work, tags, sentiment y content.
    """
    def post(self, request, *args, **kwargs):
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        import json
        import re
        
        # Obtener parámetros
        input_text = request.data.get("input_text")
        model = request.data.get("model", "deepseek-r1:32b")
        temperature = float(request.data.get("temperature", 0.7))
        
        if not input_text:
            return Response(
                {"error": "Se requiere el texto de entrada."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Construir el prompt para solicitar específicamente los campos requeridos
            system_prompt = """
            Eres un asistente analítico. Tu respuesta debe ser únicamente un JSON válido sin comentarios adicionales.
            Analiza el texto proporcionado y extrae la siguiente información:
            
            Responde con un JSON EXACTO con este esquema:
            {
            "  \"author\": \"Autor (completa si es parcial, por ejemplo, 'Pascal' se transforma en 'Blaise Pascal')\",\n"
              "title": "Título del contenido (si existe)",
            "  \"work\": \"Obra o fuente, o una cadena vacía\",\n"
              "tags": ["etiqueta1", "etiqueta2", ...],
              "sentiment_word": "Análisis de sentimiento (positivo, negativo, neutral)",
            "  \"languages\": [\"Código de idioma (ej. 'es', 'en', 'fr', etc.)\"],\n"

            "  \"sentiment_value\": \"Un número entre -1 y 1, donde -1 es extremadamente negativo, 0 es neutral y 1 es extremadamente positivo. Devuelve solamente el número.\",\n"
            "  \"analysis\": \"Un análisis profundo del contenido.\",\n"
            "  \"content_type\": \"Tipo de contenido (por ejemplo, 'artículo', 'cita', etc.)\",\n"

              "content": "El texto original limpio"
            }
            
            No incluyas explicaciones adicionales, solo el JSON.
            "Ejemplo 1:\n"
            "{\n"
            "  \"title\": \"El poder del cambio\",\n"
            "  \"tags\": [\"motivación\", \"cambio\"],\n"
            "  \"author\": \"Gabriel García Márquez\",\n"
            "  \"work\": \"Cuentos Cortos\",\n"
            "  \"languages\": [\"es\"],\n"
            "  \"sentiment_word\": \"positivo\",\n"
            "  \"sentiment_value\": 0.7,\n"
            "  \"analysis\": \"Inspira reflexión y destaca la transformación personal.\",\n"
            "  \"categories\": [\"Inspiración\", \"Reflexión\"],\n"
            "  \"keywords\": [\"cambio\", \"transformación\"],\n"
            "  \"content_type\": \"artículo\",\n"
            "  \"multilingual\": false,\n"
            "  \"content\": \"El texto explora cómo los cambios en la vida pueden abrir nuevas oportunidades.\"\n"
            "}\n\n"
            "Ejemplo 2:\n"
            "{\n"
            "  \"title\": \"\",\n"
            "  \"tags\": [\"cita\", \"motivación\"],\n"
            "  \"author\": \"Nelson Mandela\",\n"
            "  \"work\": \"\",\n"
            "  \"languages\": [\"en\"],\n"
            "  \"sentiment_word\": \"positivo\",\n"
            "  \"sentiment_value\": 0.9,\n"
            "  \"analysis\": \"Cita que enfatiza la perseverancia ante la adversidad.\",\n"
            "  \"content_type\": \"cita\",\n"
            "  \"multilingual\": false,\n"
            "  \"content\": \"It always seems impossible until it's done.\"\n"
            "}\n"
            """
            
            # Crear los mensajes para el modelo
            messages = [
                HumanMessage(content=system_prompt),
                HumanMessage(content=input_text)
            ]
            
            # Inicializar el modelo LLM
            llm = ChatOllama(
                base_url="http://localhost:11434",
                model=model,
                temperature=temperature
            )
            
            # Ejecutar el modelo
            result = llm.invoke(messages)
            response_str = result.content if hasattr(result, "content") else result
            
            # Extraer el JSON de la respuesta
            def extract_json_from_response(text):
                # Primero intentar encontrar JSON entre ```json y ```
                md_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
                md_match = md_pattern.search(text)
                if md_match:
                    json_str = md_match.group(1)
                    try:
                        return json.loads(json_str)
                    except:
                        pass
                
                # Si no se encuentra, intentar extraer el primer objeto JSON en el texto
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and start < end:
                    possible_json = text[start:end+1].strip()
                    try:
                        return json.loads(possible_json)
                    except:
                        return None
                return None
            
            parsed_result = extract_json_from_response(response_str)
            
            if not parsed_result:
                return Response(
                    {"error": "Error al parsear la respuesta JSON", "raw": response_str},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Asegurar que todos los campos solicitados estén presentes
            required_fields = ["author", "title", "work", "tags", "sentiment", "content", "content_type", "multilingual", "languages", "sentiment_word", "sentiment_value", "analysis"]
            for field in required_fields:
                if field not in parsed_result:
                    parsed_result[field] = "" if field != "tags" else []
            
            return Response(parsed_result, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class TextMetadataProcessingView(APIView):
    """
    Endpoint para guardar los metadatos de textos procesados.
    
    POST: Recibe un array de metadatos de textos y crea archivos JSON en la carpeta apropiada,
    genera embeddings y crea nodos UnconnectedDoc en Neo4j para cada texto.
    Los embeddings se almacenan en Weaviate, mientras que los metadatos se guardan en Neo4j.
    
    Parámetros esperados:
    Array de objetos con los metadatos obtenidos del endpoint TextProcessView, incluyendo:
    - author: Autor del texto
    - title: Título del contenido
    - work: Obra o fuente
    - tags: Array de etiquetas
    - sentiment_word: Análisis de sentimiento en palabras (positivo, negativo, neutral)
    - sentiment_value: Valor numérico de sentimiento (-1 a 1)
    - analysis: Análisis profundo del contenido
    - content: El texto original
    """
    def post(self, request, *args, **kwargs):
        from langchain_ollama import OllamaEmbeddings
        import json
        import os
        import uuid
        
        try:
            # Verificar si recibimos un array o un objeto único
            metadata_list = request.data
            if not isinstance(metadata_list, list):
                metadata_list = [metadata_list]  # Convertir a lista si es un objeto único
            
            if not metadata_list:
                return Response(
                    {"error": "Se requieren metadatos en formato JSON."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Inicializar modelo de embeddings
            embedding_model = OllamaEmbeddings(model="granite-embedding:latest")
            results = []
            
            for metadata in metadata_list:
                if not metadata or not isinstance(metadata, dict):
                    results.append({
                        "error": "Un elemento del array no es un objeto JSON válido."
                    })
                    continue
                
                # Verificar que el contenido esté presente
                if not metadata.get("content"):
                    results.append({
                        "error": "El campo 'content' es requerido."
                    })
                    continue
                
                # Generar un UUID único
                doc_id = str(uuid.uuid4())
                metadata["id"] = doc_id
                metadata["doc_id"] = doc_id
                
                # Establecer content_type si no está definido
                if not metadata.get("content_type"):
                    metadata["content_type"] = "text"
                
                # Asegurar que los campos requeridos estén presentes
                required_fields = ["author", "title", "work", "tags", "sentiment_word", 
                                  "sentiment_value", "analysis", "content"]
                for field in required_fields:
                    if field not in metadata:
                        if field == "tags":
                            metadata[field] = []
                        elif field == "sentiment_value":
                            metadata[field] = 0
                        else:
                            metadata[field] = ""
                
                # Generar embeddings
                embeddings = embedding_model.embed_documents([metadata.get("content", "")])
                
                # Determinar la ubicación del archivo
                uploads_dir = os.path.join('uploads', 'texts')
                if not os.path.exists(uploads_dir):
                    os.makedirs(uploads_dir)
                
                # Guardar el archivo JSON
                json_filename = f"{doc_id}.json"
                json_path = os.path.join(uploads_dir, json_filename)
                file_location = f"uploads/texts/{json_filename}"
                metadata["file_location"] = file_location
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # Crear nodo en Neo4j y guardar embedding en Weaviate
                node = store_embedding(
                    doc_id=doc_id,
                    embedding=embeddings[0] if embeddings else [],
                    label="UnconnectedDoc",
                    meta=metadata
                )
                
                results.append({
                    "status": "OK",
                    "doc_id": doc_id,
                    "file_location": file_location,
                    "message": "Metadatos guardados y nodo creado exitosamente."
                })
            
            return Response({
                "status": "OK",
                "results": results
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )