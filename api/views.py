# api/views.py
from .embeddings_to_neo import store_embedding
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import UploadedFileSerializer
import os
import json
import shutil
from django.conf import settings
from rest_framework import status
from .models import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

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
                "location": instance.file.url,  # URL del archivo almacenado
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
                    
                # uploaded_file.status = "categorized"
                # uploaded_file.save()
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
