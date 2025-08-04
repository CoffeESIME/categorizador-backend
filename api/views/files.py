import os
import json
import uuid

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from langchain_ollama import OllamaEmbeddings

from ..embeddings_to_neo import store_embedding
from ..serializers import UploadedFileSerializer
from ..models import UploadedFile

from ..services.processing import (
    process_pdf,
    process_image_with_description,
    process_ocr_with_image,
    process_text_embeddings,
    process_audio_file,
    process_video_file,
)
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
                print("data", meta, embedding_type)
                try:
                    if embedding_type == "pdf":
                        process_pdf(file_id, meta)

                    elif embedding_type == "image_w_des":
                        image_path = uploads_dir
                        if not image_path or not os.path.exists(image_path):
                            results.append({"id": file_id, "error": "No se encontró la imagen en 'file_location'"})
                            continue
                        process_image_with_description(file_id, meta, image_path)

                    elif embedding_type == "ocr_w_img":
                        image_path = uploads_dir
                        if not image_path or not os.path.exists(image_path):
                            results.append({"id": file_id, "error": "No se encontró la imagen en 'file_location'"})
                            continue
                        process_ocr_with_image(file_id, meta, image_path)

                    elif embedding_type == "audio":
                        audio_path = uploads_dir
                        if not audio_path or not os.path.exists(audio_path):
                            results.append({"id": file_id, "error": "No se encontró el audio en 'file_location'"})
                            continue
                        process_audio_file(file_id, meta, audio_path)
                        embedding = [1]

                    elif embedding_type in ["audio_text", "audio+text"]:
                        audio_path = uploads_dir
                        if not audio_path or not os.path.exists(audio_path):
                            results.append({"id": file_id, "error": "No se encontró el audio en 'file_location'"})
                            continue
                        process_audio_file(file_id, meta, audio_path, transcribe=True)
                        embedding = [1]

                    elif embedding_type in ["video_audio", "video+audio"]:
                        video_path = uploads_dir
                        if not video_path or not os.path.exists(video_path):
                            results.append({"id": file_id, "error": "No se encontró el video en 'file_location'"})
                            continue
                        process_video_file(file_id, meta, video_path, include_audio=True)
                        embedding = [1]

                    elif embedding_type == "video":
                        video_path = uploads_dir
                        if not video_path or not os.path.exists(video_path):
                            results.append({"id": file_id, "error": "No se encontró el video en 'file_location'"})
                            continue
                        print('now here')
                        process_video_file(file_id, meta, video_path)
                        embedding = [1]

                    else:
                        embedding = process_text_embeddings(meta)

                except Exception as proc_err:
                    results.append({"id": file_id, "error": str(proc_err)})
                    continue
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
                base_url=settings.LLM_BASE_URL,
                model=model,
                temperature=temperature,
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
            embedding_model = OllamaEmbeddings(
                model=settings.DEFAULT_EMBED_MODEL,
                base_url=settings.LLM_BASE_URL,
            )
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
            
            
            
            
            
