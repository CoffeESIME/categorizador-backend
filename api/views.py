import re
import traceback
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
                file_id = meta.get("original_name")
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

# ... (dentro de GraphView) ...
            for record in result:
                node_data = record.get("n")
                node = {}
                if node_data:
                    for key, value in node_data.items():
                        # Convertir tipos de fecha/hora de Neo4j a string
                        if hasattr(value, 'isoformat'): # Verifica si es un objeto de fecha/hora compatible
                            node[key] = value.isoformat()
                        else:
                            node[key] = value
                
                # Usamos 'doc_id' si existe, o 'id' como respaldo para identificar el nodo
                node_id = node.get("doc_id") or node.get("id")
                node["nodeId"] = node_id # Asegurarse de que nodeId se asigna después de procesar las propiedades
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
                    a_data = record.get("a")
                    b_data = record.get("b")
                    
                    a_node = {}
                    if a_data:
                        for key, value in a_data.items():
                            if hasattr(value, 'isoformat'):
                                a_node[key] = value.isoformat()
                            else:
                                a_node[key] = value
                                
                    b_node = {}
                    if b_data:
                        for key, value in b_data.items():
                            if hasattr(value, 'isoformat'):
                                b_node[key] = value.isoformat()
                            else:
                                b_node[key] = value
                    
                    source = a_node.get("doc_id") or a_node.get("id")
                    target = b_node.get("doc_id") or b_node.get("id")
                    edges.append({
                        "source": source,
                        "target": target,
                        "relation": record.get("relation")
                    })
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


from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
from .neo4j_client import driver 

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

class AdvancedGraphSearchView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            params = request.data
            
            # Validar parámetros básicos (puedes usar un Serializer de DRF para esto)
            # Por ahora, una validación simple
            if not isinstance(params, dict):
                return Response({"error": "El cuerpo de la solicitud debe ser un objeto JSON."},
                                status=status.HTTP_400_BAD_REQUEST)

            # --- Construcción de la Consulta Cypher ---
            # Esta parte será compleja y requerirá construir la consulta dinámicamente
            # basándose en los parámetros.
            
            cypher_query_parts = []
            cypher_params = {}
            
            # 1. Nodos de Inicio (MATCH inicial)
            # Esta es la parte más compleja de generalizar. Podrías empezar con un MATCH (n) general
            # y luego filtrar, o construir cláusulas MATCH más específicas si hay start_nodes.
            # Por simplicidad, aquí asumiremos un inicio más general o un filtrado posterior.
            
            # Ejemplo de manejo de start_nodes (simplificado):
            # Si se proveen start_nodes, podrías hacer un UNWIND y MATCH
            # O construir un WHERE con ORs para los IDs/propiedades de los nodos de inicio.

            # 2. Criterios de Coincidencia para Nodos
            node_alias = "n" # Alias para los nodos principales en el path
            start_node_alias = "start_node"
            end_node_alias = "end_node" # Alias para los nodos finales del path
            rel_alias = "r" # Alias para las relaciones

            # MATCH clause
            # Determinar la profundidad del path
            min_depth = params.get("traversal_options", {}).get("min_depth", 0)
            max_depth = params.get("traversal_options", {}).get("max_depth", 3)
            
            path_pattern = f"({start_node_alias})-[{rel_alias}*"""
            if min_depth is not None and max_depth is not None:
                 path_pattern += f"{min_depth}..{max_depth}"
            elif max_depth is not None:
                 path_pattern += f"..{max_depth}"
            else: # Solo min_depth o ninguno (usa un default razonable)
                 path_pattern += f"{min_depth}.." # O simplemente * si min_depth es 0
            path_pattern += f"]-({end_node_alias})"

            cypher_query_parts.append(f"MATCH path = {path_pattern}")

            # WHERE clause
            where_clauses = []

            # Filtrado de Nodos de Inicio (start_node_alias)
            start_nodes_criteria = params.get("start_nodes", [])
            if start_nodes_criteria:
                start_node_conditions = []
                for i, criteria in enumerate(start_nodes_criteria):
                    alias_param = f"start_crit_{i}"
                    condition_parts = []
                    if "id" in criteria:
                        condition_parts.append(f"{start_node_alias}.id = ${alias_param}_id")
                        cypher_params[f"{alias_param}_id"] = criteria["id"]
                    if "label" in criteria:
                        condition_parts.append(f"'{criteria['label']}' IN labels({start_node_alias})")
                    if "properties" in criteria:
                        for p_key, p_value in criteria["properties"].items():
                            prop_param_name = f"{alias_param}_prop_{p_key}"
                            condition_parts.append(f"{start_node_alias}.`{p_key}` = ${prop_param_name}")
                            cypher_params[prop_param_name] = p_value
                    if condition_parts:
                         start_node_conditions.append(f"({' AND '.join(condition_parts)})")
                if start_node_conditions:
                    where_clauses.append(f"({' OR '.join(start_node_conditions)})")


            # Filtrado de Nodos en el Path (n) - se aplica a start_node y end_node
            match_criteria = params.get("match_criteria", {})
            node_labels = match_criteria.get("node_labels", [])
            if node_labels:
                # Aplicar a ambos nodos del path (o a todos los nodos si UNWIND)
                for alias in [start_node_alias, end_node_alias]:
                    label_conditions = [f"'{label}' IN labels({alias})" for label in node_labels]
                    if label_conditions: # Si solo algunas etiquetas aplican a un tipo de nodo
                        where_clauses.append(f"({' OR '.join(label_conditions)})")


            node_prop_filters = match_criteria.get("node_properties_filter", [])
            if node_prop_filters:
                for alias in [start_node_alias, end_node_alias]: # Aplicar a ambos extremos del path
                    for i, prop_filter in enumerate(node_prop_filters):
                        filter_param_name = f"{alias}_node_prop_filter_{i}"
                        # Cuidado con la inyección de Cypher si 'key' u 'operator' vienen del usuario sin sanear
                        # Es mejor tener un mapeo de operadores seguros
                        operator = prop_filter.get("operator", "=")
                        # Validar/mapear operadores aquí para seguridad
                        safe_operators = {
                            "=": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
                            "CONTAINS": "CONTAINS", "STARTS WITH": "STARTS WITH", "ENDS WITH": "ENDS WITH",
                            "IN": "IN"
                        }
                        if operator not in safe_operators:
                            continue # O lanzar error
                        
                        where_clauses.append(f"{alias}.`{prop_filter['key']}` {safe_operators[operator]} ${filter_param_name}")
                        cypher_params[filter_param_name] = prop_filter['value']
            
            # Filtrado de Relaciones (r)
            # Esto requiere iterar sobre las relaciones en el path si quieres filtrar cada una.
            # UNWIND relationships(path) as r_in_path
            # WITH start_node, end_node, r_in_path, path
            # WHERE ...
            # Esto añade complejidad. Por ahora, un filtro general para los tipos de relación:
            rel_types = match_criteria.get("relationship_types", [])
            if rel_types:
                 # Esta condición se aplicaría a CUALQUIER relación en el path si es *
                 # Si el path es de longitud fija, puedes ser más específico.
                 # Para paths de longitud variable, se necesita UNWIND o ALL/ANY en predicados.
                rel_type_conditions = [f"type({rel_alias_in_path}) = '{rt}'" for rt in rel_types]
                # Esto necesitaría un `UNWIND relationships(path) AS rel_alias_in_path` y aplicar el WHERE
                # O usar un predicado:
                if rel_types:
                    rel_type_check = " AND ".join([f"ANY(r_in_path IN relationships(path) WHERE type(r_in_path) = '{rt}')" for rt in rel_types])
                    # Esto es más complejo de lo que parece para múltiples tipos opcionales.
                    # Una forma más simple si quieres que *todas* las relaciones sean de ciertos tipos:
                    # where_clauses.append(f"ALL(r_in_path IN relationships(path) WHERE type(r_in_path) IN $rel_types_param)")
                    # cypher_params["rel_types_param"] = rel_types
                    # O si *alguna* relación debe ser de un tipo (lo cual es raro para un path entero):
                    # where_clauses.append(f"ANY(r_in_path IN relationships(path) WHERE type(r_in_path) IN $rel_types_param)")
                    # Por ahora, simplifiquemos: si se especifica un tipo, asumimos que se aplica a *alguna* relacion del path
                    # (esto puede no ser lo que se quiere siempre)
                    # Si la longitud del path es 1, entonces es type(r)
                    if min_depth == 1 and max_depth == 1:
                        rel_type_conditions_single = [f"type({rel_alias}) = '{rt}'" for rt in rel_types]
                        if rel_type_conditions_single:
                            where_clauses.append(f"({' OR '.join(rel_type_conditions_single)})")
                    # Para paths variables, filtrar tipos de relación es más complejo y usualmente se hace
                    # con UNWIND o con funciones de lista de predicados.

            # Dirección de la relación (ya implícita en el MATCH start-[...]-end)
            # Si quieres cambiar la direccionalidad dinámicamente, la construcción del MATCH cambia.

            if where_clauses:
                cypher_query_parts.append("WHERE " + " AND ".join(where_clauses))

            # RETURN clause
            # Desestructurar el path para obtener nodos y relaciones individualmente
            cypher_query_parts.append("WITH nodes(path) AS path_nodes, relationships(path) AS path_rels")
            cypher_query_parts.append("UNWIND path_nodes AS n_in_path")
            cypher_query_parts.append("UNWIND path_rels AS r_in_path")
            
            # Aplicar filtros de propiedades a r_in_path si es necesario aquí
            # ...
            
            # Decidir qué retornar
            result_options = params.get("result_options", {})
            return_nodes_str = "RETURN DISTINCT n_in_path" # Por defecto, todos los nodos del path
            return_rels_str = ", COLLECT(DISTINCT r_in_path) AS relationships" if result_options.get("return_edges", True) else ""
            
            # Proyección de propiedades de nodo
            # node_props_to_return = result_options.get("node_properties_to_return")
            # if node_props_to_return:
            #     prop_map = ", ".join([f"{p}: n_in_path.`{p}`" for p in node_props_to_return])
            #     return_nodes_str = f"RETURN DISTINCT n_in_path {{ .id, labels: labels(n_in_path), {prop_map} }}"
            # else: # Retornar todas las propiedades (o las serializadas)
            #     return_nodes_str = f"RETURN DISTINCT n_in_path"


            # cypher_query_parts.append(f"{return_nodes_str} {return_rels_str}")
            # Esta estructura de RETURN es un poco compleja para combinar nodos y relaciones de forma distintiva.
            # Usualmente se retornan nodos y relaciones por separado o como un path.
            # Para la visualización:
            cypher_query_parts.append("RETURN DISTINCT n_in_path, r_in_path")


            # Límites
            # El límite se aplica al final. Puede ser complejo aplicar límites separados a nodos y relaciones
            # de esta manera sin subconsultas.
            # limit_nodes = result_options.get("limit_nodes", 100)
            # limit_edges = result_options.get("limit_edges", 200) # No es trivial aplicarlo directamente aquí
            # cypher_query_parts.append(f"LIMIT {limit_nodes}") # Esto limitaría el número total de filas n_in_path, r_in_path

            final_query = "\n".join(cypher_query_parts)
            print(f"Generated Cypher: {final_query}") # Reemplazo de self.stdout.write
            print(f"Parameters: {cypher_params}")

            nodes = {} # Usar un diccionario para evitar duplicados por ID
            edges = []
            
            with driver.session() as session:
                result = session.run(final_query, **cypher_params)
                for record in result:
                    node_data = record.get("n_in_path")
                    if node_data:
                        processed_node_props = serialize_neo4j_node_properties(node_data.get('_properties', {})) # _properties es donde neo4j.Result anida las props
                        node_id = node_data.get('id') # o node_data.element_id si usas IDs internos
                        
                        # Si 'id' no está en las propiedades, intenta obtenerlo del elemento mismo
                        if not node_id and hasattr(node_data, 'id'): # id del objeto nodo (entero)
                             node_id = str(node_data.id) 
                        elif not node_id and 'id' in processed_node_props: # id de tus propiedades
                             node_id = processed_node_props['id']


                        if node_id and node_id not in nodes:
                            nodes[node_id] = {
                                "id": node_id, # Asegúrate que este es el ID que usas para identificar unívocamente
                                "labels": list(node_data.labels),
                                "properties": processed_node_props
                            }

                    rel_data = record.get("r_in_path")
                    if rel_data:
                        # Para relaciones, los IDs de start y end son internos de Neo4j
                        # Necesitas mapearlos a tus IDs de propiedad si los usas para la visualización.
                        # Esto es más fácil si buscas los nodos conectados por sus propiedades `id`
                        start_node_element_id = str(rel_data.start_node.id) # ID interno del nodo Neo4j
                        end_node_element_id = str(rel_data.end_node.id) # ID interno del nodo Neo4j
                        
                        # Necesitarías una forma de mapear estos IDs internos a tus IDs de propiedad ('id' o 'doc_id')
                        # si no los tienes ya en `nodes`.
                        # O, mejor, la consulta debería devolver los IDs de propiedad de los nodos de la relación.
                        # Esto se complica. Una forma más simple es que el front-end reconstruya desde la lista de nodos.

                        # Por ahora, guardamos la relación con los IDs de los nodos que ya deberíamos tener en `nodes`
                        # Esto asume que los nodos de la relación ya fueron procesados.
                        # Se necesitará obtener el 'id' de propiedad de start_node y end_node.
                        # La consulta actual no facilita esto directamente en el `r_in_path` solo.

                        # Una mejor forma de retornar para visualización:
                        # RETURN n, r, m (nodo_inicio, relacion, nodo_fin)
                        # Luego procesas n, r, m en cada record.

                        # Replantear el RETURN y el procesamiento para que sea más fácil construir el grafo:
                        # MATCH (n1)-[r]->(n2) WHERE ... RETURN n1, r, n2
                        # Esto es mucho más simple si los filtros se pueden aplicar así.
                        
                        # Por ahora, vamos a omitir el procesamiento de relaciones detallado
                        # ya que la consulta actual con UNWIND lo hace complicado de reconstruir sin
                        # información adicional o una estructura de query diferente.
                        # La `GraphView` original tiene un enfoque más directo para esto.
                        pass


            # La consulta actual devuelve n_in_path y r_in_path, que no es ideal para reconstruir un grafo
            # de nodos y aristas únicos.
            # Vamos a simplificar la consulta para que se parezca más a tu GraphView original
            # y luego añadimos filtros.

            # --- REVISIÓN DE LA CONSTRUCCIÓN DE LA CONSULTA (Enfoque más simple) ---
            
            nodes_map = {}
            edges_list = []
            
            match_parts = []
            where_conditions = []
            query_parameters = {}

            start_nodes_config = params.get("start_nodes", [])
            path_min_depth = params.get("traversal_options", {}).get("min_depth", 0)
            path_max_depth = params.get("traversal_options", {}).get("max_depth", 2) 
            result_options = params.get("result_options", {}) # <--- Asegúrate de que result_options se defina aquí
            match_criteria = params.get("match_criteria", {}) # <--- Asegúrate de que match_criteria se defina aquí

            
            # Construir el patrón de path
            # (start_node_alias)-[rel_alias*min..max]-(end_node_alias)
            # Si no hay start_nodes_config, el match es más general: MATCH (n)
            if start_nodes_config:
                # Unir los nodos de inicio con OR y hacer el path desde ellos
                start_node_matches = []
                for i, sn_config in enumerate(start_nodes_config):
                    sn_alias = f"sn_{i}"
                    sn_match_parts = []
                    sn_where_parts = []
                    label_part = f":`{sn_config['label']}`" if "label" in sn_config else ""
                    sn_match_parts.append(f"({sn_alias}{label_part})")
                    
                    if "id" in sn_config:
                        param_name = f"sn_{i}_id"
                        sn_where_parts.append(f"{sn_alias}.id = ${param_name}")
                        query_parameters[param_name] = sn_config["id"]
                    if "properties" in sn_config:
                        for p_key, p_value in sn_config["properties"].items():
                            param_name = f"sn_{i}_prop_{p_key}"
                            sn_where_parts.append(f"{sn_alias}.`{p_key}` = ${param_name}")
                            query_parameters[param_name] = p_value
                    
                    path_query = f"MATCH path = ({sn_alias})-[r_path*{path_min_depth}..{path_max_depth}]-(m_path) "
                    if sn_where_parts:
                        path_query += "WHERE " + " AND ".join(sn_where_parts) + " "
                    
                    # Aplicar filtros de match_criteria a m_path y r_path aquí
                    sub_where_clauses_for_path = self._build_path_filters(
                        "m_path", "r_path", match_criteria, query_parameters, f"path_{i}_"
                    )
                    if sub_where_clauses_for_path:
                         path_query += ("AND " if sn_where_parts else "WHERE ") + " AND ".join(sub_where_clauses_for_path)


                    path_query += "RETURN nodes(path) AS path_nodes, relationships(path) AS path_rels"
                    match_parts.append(path_query)

                final_query = " UNION ".join(match_parts) # Si hay múltiples start_nodes, UNION los paths
            else:
                # Búsqueda general, no anclada a nodos de inicio específicos
                # Esto es más como tu GraphView actual, pero con más filtros.
                # MATCH (n)-[r*min..max]-(m)
                # O simplemente MATCH (n), luego MATCH (n)-[r]->(m) para obtener relaciones si es necesario
                # Para simplificar, si no hay start_nodes, buscamos nodos que cumplan criterios
                # y luego sus relaciones.

                node_filters_where = self._build_path_filters(
                    "n", None, match_criteria, query_parameters, "node_"
                ) # Solo filtros de nodo por ahora
                
                query_str = "MATCH (n) "
                if node_filters_where:
                    query_str += "WHERE " + " AND ".join(node_filters_where) + " "
                
                # Decidimos qué retornar: por ahora nodos y luego sus relaciones
                # Esto es más fácil de manejar que paths complejos para la salida de grafo JSON.
                # Primero obtenemos los nodos
                query_str_nodes = query_str + "RETURN DISTINCT n "
                limit_nodes = result_options.get("limit_nodes", 100)
                query_str_nodes += f"LIMIT {limit_nodes}"
                
                final_query = query_str_nodes # Ejecutaremos esto primero.

            if not final_query: # Si no se pudo construir una consulta (ej. start_nodes vacío y no se implementó alternativa)
                 return Response({"nodes": [], "edges": []}, status=status.HTTP_200_OK)


            print(f"Generated Cypher: {final_query}") # O logger.debug(f"Generated Cypher: {final_query}")
            # self.stdout.write(f"Parameters: {cypher_params}") se convierte en:
            print(f"Parameters: {cypher_params}") 

            collected_nodes_from_paths = {} # {neo4j_element_id: processed_node_dict}
            collected_rels_from_paths = {} # {neo4j_element_id: processed_rel_dict}

            with driver.session() as session:
                if start_nodes_config: # Lógica de paths
                    result = session.run(final_query, **query_parameters)
                    for record in result:
                        path_nodes_data = record.get("path_nodes", [])
                        path_rels_data = record.get("path_rels", [])

                        for node_data in path_nodes_data:
                            if node_data.element_id not in collected_nodes_from_paths:
                                props = serialize_neo4j_node_properties(node_data.get('_properties', {}))
                                node_render_id = props.get('id', props.get('doc_id', node_data.element_id))
                                collected_nodes_from_paths[node_data.element_id] = {
                                    "id": node_render_id,
                                    "labels": list(node_data.labels),
                                    "properties": props,
                                    "_neo4j_id": node_data.element_id # Guardar para mapeo de relaciones
                                }
                        for rel_data in path_rels_data:
                            if rel_data.element_id not in collected_rels_from_paths:
                                props = serialize_neo4j_node_properties(rel_data.get('_properties', {}))
                                collected_rels_from_paths[rel_data.element_id] = {
                                    "id": rel_data.element_id,
                                    "type": rel_data.type,
                                    "properties": props,
                                    "start_node_neo4j_id": rel_data.start_node.element_id,
                                    "end_node_neo4j_id": rel_data.end_node.element_id
                                }
                    
                    # Convertir a formato final
                    final_nodes_list = list(collected_nodes_from_paths.values())
                    final_edges_list = []
                    for rel_dict in collected_rels_from_paths.values():
                        start_node_internal_id = rel_dict["start_node_neo4j_id"]
                        end_node_internal_id = rel_dict["end_node_neo4j_id"]
                        
                        # Encontrar el ID de renderizado (el que usa el frontend)
                        source_render_id = collected_nodes_from_paths.get(start_node_internal_id, {}).get("id")
                        target_render_id = collected_nodes_from_paths.get(end_node_internal_id, {}).get("id")

                        if source_render_id and target_render_id:
                            final_edges_list.append({
                                "source": source_render_id,
                                "target": target_render_id,
                                "relation": rel_dict["type"],
                                "properties": rel_dict["properties"],
                                "_neo4j_id": rel_dict["id"]
                            })
                    
                    return Response({"nodes": final_nodes_list, "edges": final_edges_list}, status=status.HTTP_200_OK)

                else: # Lógica de búsqueda general de nodos primero, luego relaciones
                    node_results = session.run(final_query, **query_parameters)
                    temp_nodes_map = {} # usa el 'id' de propiedad como clave
                    node_neo4j_ids_for_rels = []

                    for record in node_results:
                        node_data = record.get("n")
                        if node_data:
                            props = serialize_neo4j_node_properties(node_data.get('_properties', {}))
                            node_render_id = props.get('id', props.get('doc_id', node_data.element_id))
                            
                            if node_render_id not in temp_nodes_map:
                                temp_nodes_map[node_render_id] = {
                                    "id": node_render_id,
                                    "labels": list(node_data.labels),
                                    "properties": props
                                }
                                node_neo4j_ids_for_rels.append(props.get('id', node_data.element_id)) # Usa el ID que usas en las relaciones

                    # Ahora obtener relaciones para estos nodos
                    final_edges_list = []
                    if temp_nodes_map and result_options.get("return_edges", True):
                        # Usar los IDs de propiedad para el MATCH de relaciones
                        rel_query_parameters = {"nodeDbIds": node_neo4j_ids_for_rels} 
                        rel_match_str = "MATCH (n1)-[r]-(n2) WHERE n1.id IN $nodeDbIds AND n2.id IN $nodeDbIds "
                        
                        # Aplicar filtros de relación aquí
                        rel_filters_where = self._build_path_filters(
                            None, "r", match_criteria, rel_query_parameters, "rel_"
                        )
                        if rel_filters_where:
                             rel_match_str += "AND " + " AND ".join(rel_filters_where)
                        
                        rel_match_str += "RETURN DISTINCT n1, r, n2 "
                        limit_edges = result_options.get("limit_edges", 200)
                        rel_match_str += f"LIMIT {limit_edges}"
                        print(f"Generated Cypher: {final_query}") 
                        print(f"Final Query (Stage 2 - Edges): {rel_match_str}")
                        print(f"Parameters (Edges): {rel_query_parameters}")

                        edge_results = session.run(rel_match_str, **rel_query_parameters)
                        for record in edge_results:
                            n1_data = record.get("n1")
                            r_data = record.get("r")
                            n2_data = record.get("n2")

                            if n1_data and r_data and n2_data:
                                n1_props = serialize_neo4j_node_properties(n1_data.get('_properties', {}))
                                n2_props = serialize_neo4j_node_properties(n2_data.get('_properties', {}))
                                r_props = serialize_neo4j_node_properties(r_data.get('_properties', {}))

                                source_id = n1_props.get('id', n1_props.get('doc_id', n1_data.element_id))
                                target_id = n2_props.get('id', n2_props.get('doc_id', n2_data.element_id))
                                
                                # Asegurarse de que los nodos de la relación estén en nuestro conjunto de nodos
                                if source_id in temp_nodes_map and target_id in temp_nodes_map:
                                    final_edges_list.append({
                                        "source": source_id,
                                        "target": target_id,
                                        "relation": r_data.type,
                                        "properties": r_props
                                    })
                    
                    return Response({"nodes": list(temp_nodes_map.values()), "edges": final_edges_list}, status=status.HTTP_200_OK)


        except Exception as e:
            # self.stderr.write(traceback.format_exc()) se convierte en:
            print(f"ERROR: {traceback.format_exc()}") # O logger.error(traceback.format_exc())
            return Response({"error": str(e), "trace": traceback.format_exc()},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _build_path_filters(self, node_alias, rel_alias, match_criteria, query_params, param_prefix=""):
        """Helper para construir cláusulas WHERE para nodos y relaciones en un path."""
        where_clauses = []
        
        # Filtros de etiqueta de nodo (si node_alias está presente)
        if node_alias:
            node_labels = match_criteria.get("node_labels", [])
            if node_labels:
                # Si un nodo DEBE tener TODAS las etiquetas especificadas:
                # label_conditions = [f"'{label}' IN labels({node_alias})" for label in node_labels]
                # where_clauses.append("(" + " AND ".join(label_conditions) + ")")
                # Si un nodo DEBE tener ALGUNA de las etiquetas especificadas (más común para filtro):
                label_conditions = [f"{node_alias}:`{label}`" for label in node_labels] # Usa sintaxis de match de etiqueta
                if label_conditions:
                    where_clauses.append("(" + " OR ".join(label_conditions) + ")")


        # Filtros de propiedades de nodo (si node_alias está presente)
        if node_alias:
            node_prop_filters = match_criteria.get("node_properties_filter", [])
            for i, prop_filter in enumerate(node_prop_filters):
                filter_param_name = f"{param_prefix}{node_alias}_prop_{i}"
                operator = prop_filter.get("operator", "=")
                key = prop_filter['key']
                value = prop_filter['value']
                
                # Mapeo de operadores seguros (expandir según sea necesario)
                safe_operators = {
                    "=": "=", "!=": "<>", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
                    "CONTAINS": "CONTAINS", "STARTS WITH": "STARTS WITH", 
                    "ENDS WITH": "ENDS WITH", "IN": "IN"
                }
                if operator not in safe_operators:
                    self.stderr.write(f"Operador no seguro o desconocido: {operator}")
                    continue
                
                # Para el operador IN, el valor debe ser una lista
                if safe_operators[operator] == "IN" and not isinstance(value, list):
                    self.stderr.write(f"Valor para operador IN debe ser una lista para la clave {key}")
                    continue

                where_clauses.append(f"{node_alias}.`{key}` {safe_operators[operator]} ${filter_param_name}")
                query_params[filter_param_name] = value

        # Filtros de tipo de relación (si rel_alias está presente)
        if rel_alias:
            rel_types = match_criteria.get("relationship_types", [])
            if rel_types:
                # Si la relación DEBE ser DE ALGUNO de estos tipos
                type_conditions = [f"type({rel_alias}) = '{rt}'" for rt in rel_types]
                if type_conditions:
                    where_clauses.append("(" + " OR ".join(type_conditions) + ")")
        
        # Filtros de propiedades de relación (si rel_alias está presente)
        if rel_alias:
            rel_prop_filters = match_criteria.get("relationship_properties_filter", [])
            for i, prop_filter in enumerate(rel_prop_filters):
                filter_param_name = f"{param_prefix}{rel_alias}_prop_{i}"
                operator = prop_filter.get("operator", "=")
                key = prop_filter['key']
                value = prop_filter['value']

                safe_operators = {
                    "=": "=", "!=": "<>", ">": ">", "<": "<", ">=": ">=", "<=": "<=",
                    "CONTAINS": "CONTAINS", "STARTS WITH": "STARTS WITH", 
                    "ENDS WITH": "ENDS WITH", "IN": "IN"
                } # Reutilizar el mismo mapeo
                if operator not in safe_operators:
                    self.stderr.write(f"Operador no seguro o desconocido para relación: {operator}")
                    continue
                
                if safe_operators[operator] == "IN" and not isinstance(value, list):
                     self.stderr.write(f"Valor para operador IN (relación) debe ser una lista para la clave {key}")
                     continue

                where_clauses.append(f"{rel_alias}.`{key}` {safe_operators[operator]} ${filter_param_name}")
                query_params[filter_param_name] = value
        
        return where_clauses


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