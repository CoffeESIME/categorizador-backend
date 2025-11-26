import os
import re
import json
import base64
import copy
from io import BytesIO
from typing import Any, Dict
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip
import requests

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings

# Importamos ChatOllama y HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from api.services.processing import _transcribe_audio
# al inicio del archivo (después de los imports estándar)
import logging

logger = logging.getLogger(__name__)          # usa el nombre del módulo
logger.setLevel(logging.INFO)

# evita duplicar handlers si Django ya configuró logging
if not logger.handlers:
    handler = logging.StreamHandler()         # STDERR – visible en gunicorn o runserver
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
# -----------------------------------------------------------------------------
# VIDEO PIPELINE
# -----------------------------------------------------------------------------
def pil_to_b64(img: Image.Image) -> str:
    """
    Convierte un objeto PIL.Image en una cadena base64 (JPEG).
    Devuelve: str → "iVBORw0KGgoAAA..."
    """
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



# video_llm_processor.py  (o donde tengas el helper)
# ---------------------------------------------------------------------------
# Función principal (versión que aplana el JSON)
# ---------------------------------------------------------------------------

def process_video_llm(
    file_path: str,
    *,
    frame_interval: int = 2,
    frame_prompt: str = "Describe brevemente el fotograma.",
    context: str = "",
    base_url: str,
    vision_model: str,
    analysis_model: str,
    temperature: float = 0.5,
    max_tokens: int = 2048,
) -> dict:
    """
    Devuelve un dict con:
        description, tags, topics, style, color_palette, composition, …
        frame_descriptions: [str, str, …]

    @raise FileNotFoundError, RuntimeError, ValueError
    """
    logger.info("⏩ [VIDEO] Procesando archivo=%s", file_path)

    # 1) Fotogramas
    frames = extract_frames(file_path, every_n_seconds=frame_interval)
    if not frames:
        raise RuntimeError("No se pudieron extraer fotogramas")
    logger.info("🔍 [VIDEO] %d frames extraídos", len(frames))

    # 2) Descripción de frames
    vision_llm = ChatOllama(
        base_url=base_url,
        model=vision_model,
        temperature=temperature,
        num_predict=max_tokens,
    )
    frame_desc: list[str] = []
    for f in frames:
        b64 = pil_to_b64(f)
        prompt = build_few_shot_image_prompt(frame_prompt, b64, "image")
        resp = vision_llm.invoke(prompt)
        txt = resp.content if hasattr(resp, "content") else resp
        parsed = extract_json_from_response(txt)
        frame_desc.append(parsed.get("description") if parsed else txt.strip())

    # 3) Resumen
    summary_llm = ChatOllama(
        base_url=base_url,
        model=analysis_model,
        temperature=temperature,
        num_predict=max_tokens,
    )
    summary_prompt = build_video_summary_prompt(frame_desc, context)
    raw = summary_llm.invoke(summary_prompt)
    txt = raw.content if hasattr(raw, "content") else raw
    summary_json = extract_json_from_response(txt)
    if summary_json is None:
        raise ValueError("Modelo no devolvió JSON válido para el resumen")

    logger.info("🎉 [VIDEO] Resumen listo. Devolviendo estructura aplanada")

    # 🚩 UNION: metemos el resumen al mismo nivel
    summary_json["frame_descriptions"] = frame_desc
    return summary_json


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

def build_structural_prompt(content_type: str = "generic") -> str:
    """
    Devuelve instrucciones de sistema para que TODOS los modelos respondan
    con un JSON consistente.  Agrega campos opcionales según content_type.
    """
    base = (
        "Eres un asistente analítico. TU RESPUESTA DEBE SER exclusivamente "
        "un JSON VÁLIDO sin comentarios ni texto extra. Utiliza SIEMPRE este "
        "núcleo mínimo de campos, en este orden:\n\n"
        "{\n"
        '  "description":  <string>,                # descripción principal\n'
        '  "tags":         <array[string]>,         # etiquetas clave\n'
        '  "topics":       <array[string]>,         # temas o categorías\n'
        '  "content_type": "<image|text|ocr|audio|music|video|other>"\n'
    )

    extras: dict[str, str] = {
        "image": (
            '  ,"style":         <string>,            # estilo visual\n'
            '  ,"color_palette": <array[string]>,     # opcional\n'
            '  ,"composition":   <string>             # opcional\n'
        ),
        "audio": (
            '  ,"genre":   <string>,                  # ej. "rock", "clásica"\n'
            '  ,"mood":    <string>,                  # ej. "enérgico", "melancólico"\n'
            '  ,"tempo":   <int>,                     # BPM estimados\n'
            '  ,"language":<string>                   # idioma dominante\n'
        ),
        "music": (  # alias de audio
            '  ,"genre":   <string>,\n'
            '  ,"mood":    <string>,\n'
            '  ,"tempo":   <int>,\n'
            '  ,"language":<string>\n'
        ),
        "text": (
            '  ,"sentiment_word":  <string>,          # "positivo"|…\n'
            '  ,"sentiment_value": <float>            # -1.0 … 1.0\n'
        ),
        "ocr": (  # mismos extras que text
            '  ,"sentiment_word":  <string>,\n'
            '  ,"sentiment_value": <float>\n'
        ),
    }

    # Cierra el objeto JSON correctamente
    base_close = "}\n\n"

    # Construye bloque de extras si aplica
    extra_block = extras.get(content_type, "")
    if extra_block and extra_block.startswith("  ,"):
        base = base.rstrip("\n") + extra_block + "\n"

    # Ejemplos didácticos para el modelo
    if content_type == "image":
        examples = (
            "Ejemplo 1:\n"
            "{\n"
            '  "description": "Atardecer vibrante sobre montañas.",\n'
            '  "tags": ["sunset", "mountains"],\n'
            '  "topics": ["nature"],\n'
            '  "content_type": "image",\n'
            '  "style": "Realista",\n'
            '  "color_palette": ["orange", "purple"]\n'
            "}\n\n"
            "Ejemplo 2:\n"
            "{\n"
            '  "description": "Ilustración minimalista en B/N.",\n'
            '  "tags": ["minimalism"],\n'
            '  "topics": ["art"],\n'
            '  "content_type": "image",\n'
            '  "style": "Minimalista"\n'
            "}\n"
        )
    elif content_type in ("audio", "music"):
        examples = (
            "Ejemplo 1:\n"
            "{\n"
            '  "description": "Guitarras distorsionadas y batería rápida.",\n'
            '  "tags": ["rock", "guitar"],\n'
            '  "topics": ["music"],\n'
            '  "content_type": "music",\n'
            '  "genre": "rock",\n'
            '  "mood": "enérgico",\n'
            '  "tempo": 160,\n'
            '  "language": "en"\n'
            "}\n\n"
            "Ejemplo 2:\n"
            "{\n"
            '  "description": "Piano suave con cuerdas de fondo.",\n'
            '  "tags": ["instrumental", "relax"],\n'
            '  "topics": ["wellness"],\n'
            '  "content_type": "music",\n'
            '  "genre": "clásica",\n'
            '  "mood": "calmado",\n'
            '  "tempo": 70,\n'
            '  "language": "instrumental"\n'
            "}\n"
        )
    else:
        examples = ""

    return (
        base + base_close +
        "1) NO añadas texto fuera del JSON.\n"
        "2) Sigue exactamente el esquema. Campos opcionales pueden omitirse "
        "si no aplican, pero respeta el orden.\n\n" +
        examples
    )
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


def transcribe_with_speaches(
    audio_file,
    *,
    base_url: str,
    model_id: str,
    extra_params: Dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: int = 300,
) -> tuple[str, Dict[str, Any]]:
    """Envia un archivo de audio a Speaches y devuelve la transcripcion resultante.
    Retorna el texto transcrito y la respuesta completa del servicio.
    """
    if not base_url:
        raise ValueError("No se configuro la URL base del servicio de Speaches.")
    if not model_id:
        raise ValueError("No se configuro el modelo de transcripcion para Speaches.")

    audio_file.seek(0)
    audio_bytes = audio_file.read()
    if not audio_bytes:
        raise ValueError("El archivo de audio recibido esta vacio.")

    url = base_url.rstrip("/") + "/v1/audio/transcriptions"
    files = {
        "file": (
            getattr(audio_file, "name", "audio.wav"),
            audio_bytes,
            getattr(audio_file, "content_type", None) or "audio/wav",
        )
    }
    data: Dict[str, Any] = {"model": model_id}
    if extra_params:
        for key, val in extra_params.items():
            if val is not None:
                data[key] = val

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(url, files=files, data=data, headers=headers, timeout=timeout)
    response.raise_for_status()
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError as exc:  # pragma: no cover - depende del servicio externo
        raise RuntimeError("Speaches devolvio una respuesta que no es JSON.") from exc

    text = payload.get("text")
    if not text or not isinstance(text, str):
        raise RuntimeError("Speaches no devolvio una transcripcion valida.")

    return text.strip(), payload


def merge_metadata(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina dos diccionarios de metadatos preservando la informacion original
    y agregando los nuevos valores del audio.
    """
    merged = copy.deepcopy(base)

    for key, value in updates.items():
        if value is None:
            continue

        if isinstance(value, str):
            if not value.strip():
                continue
            merged[key] = value
            continue

        if isinstance(value, list):
            if not value:
                continue
            existing = merged.get(key)
            if isinstance(existing, list):
                combined = existing + [item for item in value if item not in existing]
                merged[key] = combined
            else:
                merged[key] = list(value)
            continue

        if isinstance(value, dict):
            existing = merged.get(key)
            if isinstance(existing, dict):
                merged[key] = merge_metadata(existing, value)
            else:
                merged[key] = copy.deepcopy(value)
            continue

        merged[key] = value

    return merged


def build_audio_metadata_prompt(
    transcribed_text: str,
    existing_metadata: Dict[str, Any] | None = None,
    custom_prompt: str | None = None,
) -> list[HumanMessage]:
    """
    Construye un prompt que combina metadatos existentes con la transcripcion de audio.
    Devuelve instrucciones para que el LLM genere un JSON estructurado coherente.
    """
    system_message = HumanMessage(content=build_structural_prompt("audio"))

    base_instructions = [
"Eres un asistente analítico y ESTRICTO. TU RESPUESTA DEBE SER "
        "EXCLUSIVAMENTE UN ÚNICO OBJETO JSON VÁLIDO, sin texto adicional, "
        "sin comas finales y sin comentarios.\n\n"
        "Objetivo: producir un objeto que cumpla EXACTAMENTE con la interfaz "
        "`ProcessingFileMetadata` y que fusione metadatos existentes con la transcripción de audio.\n\n"
        "Usa SOLO estas claves y en ESTE ORDEN (omite las que no apliquen, pero respeta el orden):\n"
        "1. embedding_type (string)\n"
        "2. id (string)\n"
        "3. author (string, opcional)\n"
        "4. title (string, opcional)\n"
        "5. content (string, opcional)\n"
        "6. tags (array[string])\n"
        "7. sourceType (string, opcional)\n"
        "8. processingStatus (\"pending\"|\"processing\"|\"completed\"|\"failed\")\n"
        "9. processingMethod (string, opcional)\n"
        "10. llmErrorResponse (string, opcional)\n"
        "11. work (string, opcional)\n"
        "12. languages (array[string], opcional)\n"
        "13. sentiment_word (string, opcional)\n"
        "14. sentiment_value (number, opcional; -1.0..1.0)\n"
        "15. analysis (string, opcional)\n"
        "16. categories (array[string], opcional)\n"
        "17. keywords (array[string], opcional)\n"
        "18. content_type (string, opcional)\n"
        "19. multilingual (boolean, opcional)\n"
        "20. description (string, opcional)\n"
        "21. topics (array[string], opcional)\n"
        "22. style (string, opcional)\n"
        "23. color_palette (array[string], opcional)\n"
        "24. frame_descriptions (array[string], opcional)\n"
        "25. composition (string, opcional)\n"
        "26. file_type (string, opcional)\n\n"
        "Reglas:\n"
        "- SALIDA: un solo objeto JSON. Nada fuera del objeto.\n"
        "- ESTRUCTURA/ORDEN: exactamente las claves listadas y en ese orden. No inventes claves nuevas.\n"
        "- FUSIÓN: si hay metadatos iniciales, conserva lo existente cuando la transcripción no aporte cambios; "
        "actualiza o añade detalles nuevos derivados del audio.\n"
        "- DEDUPLICACIÓN: elimina duplicados en arrays (tags, languages, categories, keywords, topics, color_palette, frame_descriptions) "
        "preservando el orden de primera aparición.\n"
        "- AUDIO: coloca la transcripción íntegra en `content`; establece `content_type`=\"audio\"; si puedes, infiere `languages` y `multilingual`.\n"
        "- RESUMEN/ANÁLISIS: `description` = resumen (1–2 frases). `analysis` = puntos clave breves.\n"
        "- ESTADO: si hay transcripción utilizable, `processingStatus`=\"completed\"; si sigue en curso, \"processing\"; "
        "si solo se creó la tarea, \"pending\"; ante fallo, \"failed\" y rellena `llmErrorResponse`.\n"
        "- TIPOS: respeta los tipos. `sentiment_value` en [-1, 1].\n"
        "- IDIOMA: responde en el idioma dominante; si es mixto, usa español.\n"
        "- Si falta `id`, genera un placeholder estable tipo \"temp-<cadena>\".\n"
  
    ]

    if custom_prompt:
        base_instructions.append(f"Instrucciones adicionales del usuario: {custom_prompt}")

    prompt_sections = ["\n".join(base_instructions)]

    if existing_metadata:
        metadata_str = json.dumps(existing_metadata, ensure_ascii=False, indent=2)
        prompt_sections.append("Metadatos iniciales proporcionados por el usuario:")
        prompt_sections.append(metadata_str)

    prompt_sections.append("Transcripcion del audio (texto literal):")
    prompt_sections.append(transcribed_text.strip())

    user_message = HumanMessage(content="\n\n".join(prompt_sections))

    return [system_message, user_message]

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
    - audio: Transcribe un archivo de audio y analiza el texto resultante con un LLM.
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
                    return Response({"error": "El campo file_url es requerido."}, 400)

                file_name = os.path.basename(file_url)
                file_path = os.path.join(settings.MEDIA_ROOT, "videos", file_name)

                try:
                    result = process_video_llm(
                        file_path,
                        frame_interval = int(request.data.get("frame_interval", 2)),
                        frame_prompt   = request.data.get("prompt", "Describe brevemente el fotograma."),
                        context        = request.data.get("context", ""),
                        base_url       = settings.LLM_BASE_URL,
                        vision_model   = request.data.get("model", settings.IMAGE_DESCRIPTION_MODEL),
                        analysis_model = request.data.get("analysis_model", settings.DEFAULT_LLM_MODEL),
                        temperature    = float(request.data.get("temperature", 0.5)),
                        max_tokens     = int(request.data.get("max_tokens", 2048)),
                    )
                    return Response(result, 200)

                except Exception as exc:
                    logger.exception("💥 [VIDEO] Error procesando video")  # traza completa
                    return Response({"error": str(exc)}, 500)

            elif task == "audio":
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


class AudioMetadataCuratorView(APIView):
    """
    Endpoint que recibe un archivo de audio obligatorio y un JSON opcional de metadatos.
    Usa Speaches para transcribir el audio y un LLM para curar o complementar los metadatos.
    """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return Response(
                {"error": "El campo 'audio' es obligatorio y debe enviarse como archivo."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        metadata_raw = request.data.get("metadata")
        base_metadata: Dict[str, Any] | None = None
        if metadata_raw:
            if isinstance(metadata_raw, dict):
                base_metadata = metadata_raw
            else:
                try:
                    base_metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    return Response(
                        {"error": "El campo 'metadata' debe ser un objeto JSON v��lido."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            if base_metadata is not None and not isinstance(base_metadata, dict):
                return Response(
                    {"error": "El campo 'metadata' debe representar un objeto JSON."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        transcription_opts_raw = request.data.get("transcription_options")
        extra_params: Dict[str, Any] | None = None
        if transcription_opts_raw:
            if isinstance(transcription_opts_raw, dict):
                extra_params = transcription_opts_raw
            else:
                try:
                    extra_params = json.loads(transcription_opts_raw)
                except json.JSONDecodeError:
                    return Response(
                        {"error": "El campo 'transcription_options' debe ser un objeto JSON."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            if extra_params is not None and not isinstance(extra_params, dict):
                return Response(
                    {"error": "El campo 'transcription_options' debe representar un objeto JSON."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        speaches_base_url = request.data.get("speaches_base_url") or getattr(
            settings, "SPEACHES_BASE_URL", ""
        )
        speaches_model_id = request.data.get("speaches_model_id") or getattr(
            settings, "SPEACHES_TRANSCRIPTION_MODEL", ""
        )
        speaches_api_key = request.data.get("speaches_api_key") or getattr(
            settings, "SPEACHES_API_KEY", None
        )

        try:
            transcription, speaches_payload = transcribe_with_speaches(
                audio_file,
                base_url=speaches_base_url,
                model_id=speaches_model_id,
                extra_params=extra_params,
                api_key=speaches_api_key or None,
            )
        except Exception as exc:
            logger.exception("Error al transcribir el audio con Speaches")
            return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        try:
            temperature = float(request.data.get("temperature", 0.5))
        except (TypeError, ValueError):
            return Response(
                {"error": "El campo 'temperature' debe ser numerico."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            max_tokens = int(request.data.get("max_tokens", 2048))
        except (TypeError, ValueError):
            return Response(
                {"error": "El campo 'max_tokens' debe ser un numero entero."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        llm_model = request.data.get("model", settings.DEFAULT_LLM_MODEL)
        llm_base_url = request.data.get("llm_base_url", settings.LLM_BASE_URL)
        prompt = request.data.get("prompt")

        messages = build_audio_metadata_prompt(transcription, base_metadata, prompt)

        try:
            llm = ChatOllama(
                base_url=llm_base_url,
                model=llm_model,
                temperature=temperature,
                num_predict=max_tokens,
            )
            result = llm.invoke(messages)
        except Exception as exc:
            logger.exception("Error al invocar el modelo para curar metadatos de audio")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        response_str = result.content if hasattr(result, "content") else result
        parsed_result = extract_json_from_response(response_str)

        if not parsed_result:
            return Response(
                {
                    "error": "No se pudo interpretar la respuesta del modelo.",
                    "raw": response_str,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if base_metadata:
            final_metadata = merge_metadata(base_metadata, parsed_result)
        else:
            final_metadata = parsed_result

        include_payload = str(
            request.data.get("include_transcription_payload", "false")
        ).lower() in ("1", "true", "yes")

        response_payload = copy.deepcopy(final_metadata)
        response_payload["raw_transcription"] = transcription
        response_payload["transcription_source"] = "speaches"
        if include_payload:
            response_payload["transcription_payload"] = speaches_payload

        return Response(response_payload, status=status.HTTP_200_OK)
