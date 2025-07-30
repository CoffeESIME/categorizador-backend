import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

# Importamos ChatOllama y HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from api.services.processing import _transcribe_audio


def extract_frames(video_path: str, every_n_seconds: int = 2):
    """Extrae fotogramas de un video cada ``every_n_seconds`` segundos."""
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    frames = []
    for t in range(0, duration, every_n_seconds):
        frame = clip.get_frame(t)
        frames.append(Image.fromarray(frame))
    clip.close()
    return frames

def image_to_base64(image_path):
    """
    Convierte una imagen ubicada en image_path a una cadena en base64.
    """
    try:
        with Image.open(image_path) as img:
            # Convertir la imagen a RGB si tiene transparencia (canal alfa)
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise Exception(f"Error al convertir la imagen: {str(e)}")

def build_structural_prompt(content_type="generic"):
    """
    Construye un mensaje del sistema para una tarea específica que requiere respuesta estructurada.
    Devuelve un string con instrucciones para que el modelo responda únicamente en formato JSON.
    """
    base_structure = (
        "Eres un asistente analítico. Tu respuesta debe ser únicamente un JSON válido sin comentarios adicionales. "
        "El JSON debe incluir campos relevantes según el tipo de contenido. "
        "Asegúrate de que el JSON sea válido y no contenga texto extra."
    )
    
    if content_type == "image":
        prompt = base_structure + (
            "1) Responde con un JSON EXACTO con el siguiente esquema:\n\n"
            "{\n"
            "  - \"description\": Una descripción detallada de la imagen.\n"
            "  - \"tags\": Un array de etiquetas relevantes.\n"
            "  - \"topics\": Un array de temas relacionados.\n"
            "  - \"style\": Una descripción del estilo visual de la imagen.\n"
            "  - \"color_palette\": (Opcional) Los colores predominantes en la imagen.\n"
            "  - \"composition\": (Opcional) Notas sobre la composición de la imagen.\n"
            "}\n\n"
            "2) No añadas comentarios ni texto adicional fuera del JSON final. Responde únicamente con el JSON que cumpla EXACTAMENTE el esquema indicado.\n\n"
            "Ejemplo 1:\n"
            "{\n"
            "  \"description\": \"A vibrant sunset over a mountain range with a clear sky.\",\n"
            "  \"tags\": [\"sunset\", \"mountains\", \"landscape\"],\n"
            "  \"topics\": [\"nature\", \"landscape\", \"sunset\"],\n"
            "  \"style\": \"Realistic with warm color tones\",\n"
            "  \"color_palette\": [\"orange\", \"red\", \"purple\"],\n"
            "  \"composition\": \"Balanced composition with the horizon at the lower third.\"\n"
            "}\n\n"
            "Ejemplo 2:\n"
            "{\n"
            "  \"description\": \"Una ilustración minimalista de una ciudad futurista en blanco y negro.\",\n"
            "  \"tags\": [\"futurista\", \"ciudad\", \"minimalista\"],\n"
            "  \"topics\": [\"urbanismo\", \"arte digital\"],\n"
            "  \"style\": \"Minimalista y contrastada\",\n"
            "  \"color_palette\": [\"black\", \"white\"],\n"
            "  \"composition\": \"La composición se centra en líneas geométricas y simetría.\"\n"
            "}\n"
        )
    elif content_type == "ocr":
        prompt = base_structure + (
            "Eres un sistema de extracción de información de texto. Sigue EXACTAMENTE las siguientes instrucciones y responde únicamente con un JSON válido que cumpla este esquema:\n\n"
            "1) Responde con un JSON EXACTO con el siguiente esquema:\n\n"
            "{\n"
            "  \"title\": \"Título del contenido (si existe; en caso contrario, una cadena vacía)\",\n"
            "  \"tags\": [\"Etiqueta1\", \"Etiqueta2\", ...],\n"
            "  \"author\": \"Autor (completa si es parcial, por ejemplo, 'Pascal' se transforma en 'Blaise Pascal')\",\n"
            "  \"work\": \"Obra o fuente, o una cadena vacía\",\n"
            "  \"languages\": [\"Código de idioma (ej. 'es', 'en', 'fr', etc.)\"],\n"
            "  \"sentiment_word\": \"Análisis del sentimiento en palabras (ej. 'positivo', 'negativo', 'neutral')\",\n"
            "  \"sentiment_value\": \"Un número entre -1 y 1, donde -1 es extremadamente negativo, 0 es neutral y 1 es extremadamente positivo. Devuelve solamente el número.\",\n"
            "  \"analysis\": \"Un análisis profundo del contenido.\",\n"
            "  \"categories\": [\"Categoría o tema 1\", \"Categoría o tema 2\", ...],\n"
            "  \"keywords\": [\"Palabra clave 1\", \"Palabra clave 2\", ...],\n"
            "  \"content_type\": \"Tipo de contenido (por ejemplo, 'artículo', 'cita', etc.)\",\n"
            "  \"multilingual\": false, // Si es false, incluir 'content'. Si es true, NO incluir 'content' y usar 'eng_content', 'es_content', etc.\n"
            "  \"content\": \"El texto proporcionado, limpio y sin elementos irrelevantes. Si hay fragmentos sin sentido, elimínalos.\"\n"
            "}\n\n"
            "2) No añadas comentarios ni texto adicional fuera del JSON final.\n\n"
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
            "  \"categories\": [\"Inspiración\", \"Citas\"],\n"
            "  \"keywords\": [\"perseverancia\", \"motivación\"],\n"
            "  \"content_type\": \"cita\",\n"
            "  \"multilingual\": false,\n"
            "  \"content\": \"It always seems impossible until it's done.\"\n"
            "}\n"
        )
    elif content_type == "text":
        prompt = base_structure + (
            "Eres un sistema de extracción de información de texto. Sigue EXACTAMENTE las siguientes instrucciones y responde únicamente con un JSON válido que cumpla este esquema:\n\n"
            "1) Responde con un JSON EXACTO con el siguiente esquema:\n\n"
            "{\n"
            "  \"title\": \"Título del contenido (si existe; en caso contrario, una cadena vacía)\",\n"
            "  \"tags\": [\"Etiqueta1\", \"Etiqueta2\", ...],\n"
            "  \"author\": \"Autor (completa si es parcial, por ejemplo, 'Pascal' se transforma en 'Blaise Pascal')\",\n"
            "  \"work\": \"Obra o fuente, o una cadena vacía\",\n"
            "  \"languages\": [\"Código de idioma (ej. 'es', 'en', 'fr', etc.)\"],\n"
            "  \"sentiment_word\": \"Análisis del sentimiento en palabras (ej. 'positivo', 'negativo', 'neutral')\",\n"
            "  \"sentiment_value\": \"Un número entre -1 y 1, donde -1 es extremadamente negativo, 0 es neutral y 1 es extremadamente positivo. Devuelve solamente el número.\",\n"
            "  \"analysis\": \"Un análisis profundo del contenido.\",\n"
            "  \"categories\": [\"Categoría o tema 1\", \"Categoría o tema 2\", ...],\n"
            "  \"keywords\": [\"Palabra clave 1\", \"Palabra clave 2\", ...],\n"
            "  \"content_type\": \"Tipo de contenido (por ejemplo, 'artículo', 'cita', etc.)\",\n"
            "  \"multilingual\": false, // Si es false, incluir 'content'. Si es true, NO incluir 'content' y usar 'eng_content', 'es_content', etc.\n"
            "  \"content\": \"El texto proporcionado, limpio y sin elementos irrelevantes. Si hay fragmentos sin sentido, elimínalos.\"\n"
            "}\n\n"
            "2) No añadas comentarios ni texto adicional fuera del JSON final.\n\n"
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
            "  \"categories\": [\"Inspiración\", \"Citas\"],\n"
            "  \"keywords\": [\"perseverancia\", \"motivación\"],\n"
            "  \"content_type\": \"cita\",\n"
            "  \"multilingual\": false,\n"
            "  \"content\": \"It always seems impossible until it's done.\"\n"
            "}\n"
        )

    else:
        prompt = base_structure + (
            " Incluye campos relevantes según el contexto de la información analizada. "
            "Asegúrate de incluir todos los datos que consideres importantes."
        )
    return prompt


def build_few_shot_image_prompt(prompt_text: str, image_b64: str = None, prompt_type="image"):
    """
    Construye un prompt few-shot para tareas de imagen.
    Incluye instrucciones para devolver un JSON válido con la estructura deseada.
    """
    messages = []
    system_message = HumanMessage(
        content=build_structural_prompt(prompt_type)
    )
    messages.append(system_message)
    content_parts = []
    if image_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_b64}",
        })
    content_parts.append({
        "type": "text",
        "text": prompt_text
    })
    user_message = HumanMessage(content=content_parts)
    messages.append(user_message)
    return messages

def build_text_prompt(prompt_text: str):
    """
    Construye un prompt para tareas de texto que devuelve JSON estructurado.
    """
    messages = []
    system_message = HumanMessage(
        content=build_structural_prompt("text")
    )
    messages.append(system_message)
    user_message = HumanMessage(content=prompt_text)
    messages.append(user_message)
    return messages

def build_ocr_analysis_prompt(extracted_text: str, custom_prompt: str = None):
    """
    Construye un prompt para analizar el texto extraído de una imagen.
    """
    prompt_text = custom_prompt or (
        "Analiza el siguiente texto extraído de una imagen y devuelve un JSON estructurado "
        "con campos relevantes como 'content_summary', 'key_points', 'document_type', etc."
    )
    
    full_prompt = f"{prompt_text}\n\nTexto:\n{extracted_text}"
    
    messages = []
    system_message = HumanMessage(
        content=build_structural_prompt("text")
    )
    messages.append(system_message)
    user_message = HumanMessage(content=full_prompt)
    messages.append(user_message)
    return messages

def build_audio_analysis_prompt(transcribed_text: str, custom_prompt: str | None = None):
    """Construye un prompt para analizar texto transcrito de un audio."""
    prompt_text = custom_prompt or (
        "Analiza la siguiente transcripción de audio y devuelve un JSON estructurado "
        "con información relevante."
    )

    full_prompt = f"{prompt_text}\n\nTexto:\n{transcribed_text}"

    messages = []
    system_message = HumanMessage(
        content=build_structural_prompt("text")
    )
    messages.append(system_message)
    user_message = HumanMessage(content=full_prompt)
    messages.append(user_message)
    return messages


def build_video_summary_prompt(frame_descriptions, context: str = ""):
    """Construye un prompt para resumir descripciones de fotogramas."""
    joined = "\n".join(
        f"Frame {idx+1}: {desc}" for idx, desc in enumerate(frame_descriptions)
    )
    base_prompt = (
        "Analiza las siguientes descripciones de un video y genera un resumen en JSON "
        "con campos como description, tags, topics, style, color_palette y composition."
    )
    if context:
        base_prompt += f"\nContexto adicional: {context}"
    full_prompt = f"{base_prompt}\n\n{joined}"

    messages = []
    system_message = HumanMessage(
        content=build_structural_prompt("image")
    )
    messages.append(system_message)
    user_message = HumanMessage(content=full_prompt)
    messages.append(user_message)
    return messages

def extract_json_from_response(response_str: str):
    """
    Intenta extraer un bloque de JSON de la respuesta:
    1) Busca un bloque delimitado por ```json ... ```
    2) Si no lo encuentra, intenta encontrar el primer objeto JSON en el texto.
    Retorna un dict si se logra parsear, o None en caso de fallo.
    """
    md_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    md_match = md_pattern.search(response_str)
    if md_match:
        json_str = md_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    start = response_str.find('{')
    end = response_str.rfind('}')
    if start != -1 and end != -1 and start < end:
        possible_json = response_str[start:end+1].strip()
        try:
            return json.loads(possible_json)
        except json.JSONDecodeError:
            return None
    return None

def extract_plain_text_ocr(response_str: str):
    """
    Para OCR con LLM, extrae solo el texto plano sin estructura JSON.
    Esto es necesario para el flujo donde llava extrae texto y luego deepseek lo analiza.
    """
    # Primero intenta extraer un JSON que tenga un campo 'extracted_text'
    json_data = extract_json_from_response(response_str)
    if json_data and 'extracted_text' in json_data:
        return json_data['extracted_text']
    
    # Si no hay JSON o no tiene el campo esperado, devuelve el texto limpio
    # Elimina posibles bloques de código, marcadores JSON, etc.
    cleaned_text = re.sub(r'```.*?```', '', response_str, flags=re.DOTALL)
    cleaned_text = re.sub(r'[\{\}]', '', cleaned_text)
    return cleaned_text.strip()

class LLMProcessView(APIView):
    """
    Endpoint para procesar entradas con ChatOllama.
    
    Tareas soportadas:
    - text: Procesa un prompt de texto y devuelve una respuesta estructurada en JSON.
    - image_description: Envía una imagen (convertida a base64) para obtener una descripción estructurada con campos dinámicos
      (por ejemplo, description, tags, topics, style, etc.) usando un prompt few-shot.
    - ocr: Extrae texto de una imagen. Se puede elegir entre:
         - tesseract: Extrae el texto usando Tesseract y luego lo procesa con un LLM para obtener JSON estructurado.
         - llm: Utiliza un modelo de visión (ocr_model) para extraer el texto de la imagen y, luego, un modelo de análisis
           (analysis_model) para procesar ese texto y generar una salida estructurada en JSON.
    - music: Transcribe un archivo de audio y analiza el texto resultante con un LLM.
    - video: Analiza fotogramas de un video con un modelo de visión y genera un resumen en JSON.
    
    Parámetros (JSON):
      - task: "text", "image_description", "ocr", "music" o "video" (por defecto "text")
      - temperature: (opcional) valor para la temperatura, por defecto 0.5
      - max_tokens: (opcional) cantidad máxima de tokens, por defecto 100
      - input_text: (para "text") el prompt de texto
      - file_url: (para "image_description", "ocr" y "music") la URL o ruta relativa del archivo
      - prompt: (opcional) prompt específico para "image_description", para analizar el texto extraído en OCR o la transcripción de audio
      - ocr_method: (para "ocr") "tesseract" o "llm" (por defecto "tesseract")
      - ocr_model: (para "ocr" con método "llm") modelo a usar para extraer texto de la imagen (por defecto "llava:34b")
      - analysis_model: (para "ocr" con método "llm") modelo a usar para analizar el texto extraído (por defecto "deepseek-r1:32b")
    
    Nota: Todas las respuestas están estructuradas en formato JSON.
    """
    def post(self, request):
        task = request.data.get("task", "text")
        temperature = request.data.get("temperature", 0.5)
        max_tokens = request.data.get("max_tokens", 100000)
        downloads_dir = settings.MEDIA_ROOT
        base_url = settings.LLM_BASE_URL
        
        try:
            if task == "text":
                input_text = request.data.get("input_text")
                if not input_text:
                    return Response({"error": "El campo input_text es requerido para la tarea 'text'."},
                                    status=status.HTTP_400_BAD_REQUEST)
                
                # Construir un prompt estructurado para texto
                prompt = build_text_prompt(input_text)
                llm = ChatOllama(
                    base_url=base_url,
                    model=request.data.get("model", settings.DEFAULT_LLM_MODEL),
                    temperature=temperature,
                    num_predict=max_tokens,
                )
                result = llm.invoke(prompt)
                response_str = result.content if hasattr(result, "content") else result
                parsed_result = extract_json_from_response(response_str)
                
                if not parsed_result:
                    return Response(
                        {"error": "Error al parsear la respuesta JSON.", "raw": response_str},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                return Response(parsed_result, status=status.HTTP_200_OK)
            
            elif task == "image_description":
                file_url = request.data.get("file_url")
                prompt_text = request.data.get(
                    "prompt",
                    ""
                )
                if not file_url:
                    return Response({"error": "El campo file_url es requerido para la tarea 'image_description'."},
                                    status=status.HTTP_400_BAD_REQUEST)
                file_name = os.path.basename(file_url)
                file_path = os.path.join(downloads_dir, 'images', file_name)
                if not os.path.exists(file_path):
                    return Response({"error": "Archivo no encontrado."}, status=status.HTTP_404_NOT_FOUND)
                
                image_b64 = image_to_base64(file_path)
                messages = build_few_shot_image_prompt(prompt_text, image_b64, "image")
                llm = ChatOllama(
                    base_url=base_url,
                    model=request.data.get("model", settings.IMAGE_DESCRIPTION_MODEL),
                    temperature=temperature,
                    num_predict=max_tokens,
                )
                result = llm.invoke(messages)
                response_str = result.content if hasattr(result, "content") else result
                parsed_result = extract_json_from_response(response_str)
                if not parsed_result:
                    return Response(
                        {"error": "Error al parsear la respuesta JSON.", "raw": response_str},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                return Response(parsed_result, status=status.HTTP_200_OK)

            elif task == "video":
                file_url = request.data.get("file_url")
                if not file_url:
                    return Response({"error": "El campo file_url es requerido para la tarea 'video'."},
                                    status=status.HTTP_400_BAD_REQUEST)
                file_name = os.path.basename(file_url)
                file_path = os.path.join(downloads_dir, 'videos', file_name)
                if not os.path.exists(file_path):
                    return Response({"error": "Archivo no encontrado."}, status=status.HTTP_404_NOT_FOUND)

                frame_interval = int(request.data.get("frame_interval", 2))
                frame_prompt = request.data.get("prompt", "Describe brevemente el fotograma.")
                context = request.data.get("context", "")

                frames = extract_frames(file_path, frame_interval)
                if not frames:
                    return Response({"error": "No se pudieron extraer fotogramas del video."},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                vision_model = request.data.get("model", settings.IMAGE_DESCRIPTION_MODEL)
                vision_llm = ChatOllama(
                    base_url=base_url,
                    model=vision_model,
                    temperature=temperature,
                    num_predict=max_tokens,
                )

                descriptions = []
                for frame in frames:
                    buf = BytesIO()
                    frame.convert("RGB").save(buf, format="JPEG")
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    msg = build_few_shot_image_prompt(frame_prompt, frame_b64, "image")
                    r = vision_llm.invoke(msg)
                    content = r.content if hasattr(r, "content") else r
                    parsed = extract_json_from_response(content)
                    if parsed and isinstance(parsed, dict) and parsed.get("description"):
                        descriptions.append(parsed["description"])
                    else:
                        descriptions.append(content.strip())

                summary_prompt = build_video_summary_prompt(descriptions, context)
                analysis_model = request.data.get("analysis_model", settings.DEFAULT_LLM_MODEL)
                summary_llm = ChatOllama(
                    base_url=base_url,
                    model=analysis_model,
                    temperature=temperature,
                    num_predict=max_tokens,
                )
                summary_result = summary_llm.invoke(summary_prompt)
                response_str = summary_result.content if hasattr(summary_result, "content") else summary_result
                parsed_summary = extract_json_from_response(response_str)
                if not parsed_summary:
                    return Response(
                        {"error": "Error al parsear la respuesta JSON.", "raw": response_str},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                parsed_summary["frame_descriptions"] = descriptions
                return Response(parsed_summary, status=status.HTTP_200_OK)

            elif task == "music":
                file_url = request.data.get("file_url")
                if not file_url:
                    return Response({"error": "El campo file_url es requerido para la tarea 'music'."},
                                    status=status.HTTP_400_BAD_REQUEST)
                file_name = os.path.basename(file_url)
                file_path = os.path.join(downloads_dir, 'audio', file_name)
                if not os.path.exists(file_path):
                    return Response({"error": "Archivo no encontrado."}, status=status.HTTP_404_NOT_FOUND)

                transcription = _transcribe_audio(file_path)
                analysis_prompt = request.data.get(
                    "prompt",
                    "Analiza la siguiente transcripción de audio y devuelve un JSON estructurado."
                )

                prompt = build_audio_analysis_prompt(transcription, analysis_prompt)
                llm = ChatOllama(
                    base_url=base_url,
                    model=request.data.get("model", settings.DEFAULT_LLM_MODEL),
                    temperature=temperature,
                    num_predict=max_tokens,
                )
                result = llm.invoke(prompt)
                response_str = result.content if hasattr(result, "content") else result
                parsed_result = extract_json_from_response(response_str)
                if not parsed_result:
                    return Response(
                        {"error": "Error al parsear la respuesta JSON.", "raw": response_str},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                parsed_result["raw_transcription"] = transcription
                return Response(parsed_result, status=status.HTTP_200_OK)

            elif task == "ocr":
                file_url = request.data.get("file_url")
                if not file_url:
                    return Response({"error": "El campo file_url es requerido para la tarea 'ocr'."},
                                    status=status.HTTP_400_BAD_REQUEST)
                file_name = os.path.basename(file_url)
                file_path = os.path.join(downloads_dir, 'images', file_name)
                if not os.path.exists(file_path):
                    return Response({"error": "Archivo no encontrado."}, status=status.HTTP_404_NOT_FOUND)
                
                ocr_method = request.data.get("ocr_method", "tesseract")
                if ocr_method == "tesseract":
                    with Image.open(file_path) as img:
                        extracted_text = pytesseract.image_to_string(img)
                    analysis_prompt = request.data.get(
                        "prompt",
                        "Analiza el siguiente texto extraído mediante OCR y devuelve un JSON estructurado."
                    )
                    
                    prompt = build_ocr_analysis_prompt(extracted_text, analysis_prompt)
                    analysis_model = request.data.get("model", settings.TESSERACT_ANALYSIS_MODEL)
                    llm = ChatOllama(
                        base_url=base_url,
                        model=analysis_model,
                        temperature=temperature,
                        num_predict=max_tokens
                    )
                    result = llm.invoke(prompt)
                    response_str = result.content if hasattr(result, "content") else result
                    parsed_result = extract_json_from_response(response_str)
                    
                    if not parsed_result:
                        return Response(
                            {"error": "Error al parsear la respuesta JSON del análisis de texto.", "raw": response_str},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                    return Response(parsed_result, status=status.HTTP_200_OK)
                
                elif ocr_method == "llm":
                    prompt_text = request.data.get("prompt", "Extrae todo el texto visible en esta imagen. Solo devuelve el texto extraído, sin análisis ni comentarios adicionales.")
                    image_b64 = image_to_base64(file_path)
                    ocr_prompt = build_few_shot_image_prompt(prompt_text, image_b64, "ocr")
                    ocr_model = request.data.get("ocr_model", settings.OCR_MODEL)
                    ocr_llm = ChatOllama(
                        base_url=base_url,
                        model=ocr_model,
                        temperature=temperature,
                        num_predict=max_tokens
                    )
                    ocr_result = ocr_llm.invoke(ocr_prompt)
                    ocr_response = ocr_result.content if hasattr(ocr_result, "content") else ocr_result
                    
                    extracted_text = extract_plain_text_ocr(ocr_response)
                    
                    analysis_prompt = request.data.get(
                        "analysis_prompt",
                        "Analiza el siguiente texto extraído mediante OCR y devuelve un JSON estructurado con información relevante."
                    )
                    
                    analysis_prompt_obj = build_ocr_analysis_prompt(extracted_text, analysis_prompt)
                    
                    analysis_model = request.data.get("analysis_model", settings.OCR_ANALYSIS_MODEL)
                    analysis_llm = ChatOllama(
                        base_url=base_url,
                        model=analysis_model,
                        temperature=temperature,
                        num_predict=max_tokens
                    )
                    
                    analysis_result = analysis_llm.invoke(analysis_prompt_obj)
                    analysis_response = analysis_result.content if hasattr(analysis_result, "content") else analysis_result
                    parsed_result = extract_json_from_response(analysis_response)
                    
                    if not parsed_result:
                        return Response(
                            {
                                "error": "Error al parsear la respuesta JSON del análisis de texto.",
                                "extracted_text": extracted_text,
                                "raw_analysis": analysis_response
                            },
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                    
                    parsed_result["raw_extracted_text"] = extracted_text
                    return Response(parsed_result, status=status.HTTP_200_OK)
                
                else:
                    return Response({"error": "El valor de ocr_method no es válido. Use 'tesseract' o 'llm'."},
                                    status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Tarea no soportada. Use 'text', 'image_description', 'ocr', 'music' o 'video'."},
                                status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)