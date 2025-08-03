"""
video_llm_processor.py
----------------------

Procesa un video con un modelo de visión y un modelo de análisis de
texto para obtener:
  1. Descripción de cada fotograma (cada N segundos).
  2. Un resumen estructurado en JSON del contenido del video.

Requisitos:
  - moviepy
  - pillow
  - langchain-ollama ≥ 0.2.3
  - Un servidor Ollama corriendo con los modelos que elijas.

Uso rápido
----------
>>> from video_llm_processor import process_video
>>> result = process_video(
...     "/ruta/a/video.mp4",
...     frame_interval=2,
...     frame_prompt="Describe brevemente el fotograma.",
...     context="Este video es parte de mi serie sobre montañismo."
... )
>>> print(result["summary"])           # JSON con campos description, tags…
>>> print(result["frame_descriptions"])  # Lista de textos por fotograma
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import List, Dict, Any

from PIL import Image
from moviepy.editor import VideoFileClip

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, every_n_seconds: int = 2) -> List[Image.Image]:
    """Extrae fotogramas del vídeo cada *every_n_seconds* segundos."""
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    frames: List[Image.Image] = []
    for t in range(0, duration, every_n_seconds):
        frame = clip.get_frame(t)
        frames.append(Image.fromarray(frame))
    clip.close()
    return frames


def pil_to_b64(img: Image.Image) -> str:
    """Convierte un objeto PIL a string base64 JPEG."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Construcción de prompts
# ---------------------------------------------------------------------------

def _structural_prompt(content_type: str = "image") -> str:
    """Prompt de sistema: fuerza al modelo a devolver JSON limpio."""
    base = (
        "Eres un asistente analítico. Tu respuesta debe ser únicamente un JSON "
        "válido sin comentarios adicionales. Incluye campos relevantes según el tipo."
    )
    if content_type == "image":
        return base + (
            "\nDevuelve un objeto con al menos: "
            "\"description\", \"tags\", \"topics\", \"style\", \"color_palette\", \"composition\"."
        )
    return base


def _few_shot_image_prompt(text_prompt: str, image_b64: str) -> List[HumanMessage]:
    content = [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        {"type": "text", "text": text_prompt},
    ]
    return [
        HumanMessage(content=_structural_prompt("image")),
        HumanMessage(content=content),
    ]


def _video_summary_prompt(frame_descs: List[str], context: str = "") -> List[HumanMessage]:
    joined = "\n".join(f"Frame {i+1}: {d}" for i, d in enumerate(frame_descs))
    txt = (
        "Analiza las siguientes descripciones de un vídeo y genera un resumen en JSON "
        "con los campos description, tags, topics, style, color_palette y composition.\n\n"
        f"{joined}"
    )
    if context:
        txt += f"\n\nContexto adicional: {context}"
    return [
        HumanMessage(content=_structural_prompt("image")),
        HumanMessage(content=txt),
    ]


# ---------------------------------------------------------------------------
# Parsing seguro de las respuestas
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Devuelve el primer objeto JSON válido encontrado en *text*."""
    md = _JSON_BLOCK_RE.search(text)
    if md:
        try:
            return json.loads(md.group(1))
        except json.JSONDecodeError:
            pass
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    *,
    frame_interval: int = 2,
    frame_prompt: str = "Describe brevemente el fotograma.",
    context: str = "",
    base_url: str = "http://localhost:11434",
    vision_model: str = "llava:34b",
    analysis_model: str = "deepseek-r1:14b",
    temperature: float = 0.5,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Procesa el vídeo completo y devuelve diccionario con:
      - summary: dict con los campos del JSON resumido.
      - frame_descriptions: lista de descripciones (str) por fotograma.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    # 1. Extraer fotogramas
    frames = extract_frames(video_path, frame_interval)
    if not frames:
        raise RuntimeError("No se pudieron extraer fotogramas.")

    # LLM de visión
    vision_llm = ChatOllama(
        base_url=base_url,
        model=vision_model,
        temperature=temperature,
        num_predict=max_tokens,
    )

    # 2. Describir cada fotograma
    frame_descriptions: List[str] = []
    for frame in frames:
        b64 = pil_to_b64(frame)
        messages = _few_shot_image_prompt(frame_prompt, b64)
        raw = vision_llm.invoke(messages)
        resp = raw.content if hasattr(raw, "content") else raw
        parsed = _extract_json(resp)
        desc = parsed.get("description") if parsed else resp.strip()
        frame_descriptions.append(desc)

    # 3. Resumir todo el vídeo
    summary_llm = ChatOllama(
        base_url=base_url,
        model=analysis_model,
        temperature=temperature,
        num_predict=max_tokens,
    )
    summary_messages = _video_summary_prompt(frame_descriptions, context)
    raw_summary = summary_llm.invoke(summary_messages)
    summary_text = raw_summary.content if hasattr(raw_summary, "content") else raw_summary
    summary_json = _extract_json(summary_text)

    if summary_json is None:
        raise RuntimeError(f"No se pudo parsear el resumen JSON.\nRespuesta cruda:\n{summary_text}")

    return {
        "summary": summary_json,
        "frame_descriptions": frame_descriptions,
    }


# ---------------------------------------------------------------------------
# Pequeño CLI de prueba
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # python video_llm_processor.py /ruta/video.mp4
    import argparse
    parser = argparse.ArgumentParser(description="Analiza un vídeo con modelos Ollama.")
    parser.add_argument("video", help="Ruta al archivo de vídeo")
    parser.add_argument("--interval", type=int, default=2, help="Segundos entre fotogramas")
    parser.add_argument("--context", default="", help="Contexto adicional")
    args = parser.parse_args()

    result = process_video(
        args.video,
        frame_interval=args.interval,
        context=args.context,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
