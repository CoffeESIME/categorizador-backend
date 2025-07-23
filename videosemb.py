"""
Uso:
  poetry run python videosemb.py ruta/al/video.mp4

Retorna:
  - Lista de captions (uno por frame)
  - Embedding de la descripción global (OllamaEmbeddings)
  - Embedding CLIP promedio de todos los frames
"""

import sys
import os
import io
import tempfile
import base64
from pathlib import Path
from typing import List

import torch
import clip
from moviepy import VideoFileClip          # ← import correcto
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 1. CONFIG — AJUSTA SEGÚN TUS PREFERENCIAS
EVERY_N_SECONDS = 2                      # Intervalo de muestreo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OLLAMA_BASE_URL      = "http://localhost:11434"   # Cambia si tu Ollama corre en otro puerto
OLLAMA_VISION_MODEL  = "llava:34b"                    # Modelo vision-LLM para captions
OLLAMA_TEXT_EMB_MODEL = "nomic-embed-text"        # O el que tengas por defecto
# ─────────────────────────────────────────────────────────────

# 2.  CLIP  ──────────────────────────────────────────────────
try:
    CLIP_VISUAL, CLIP_PREPROC = clip.load("ViT-B/32", device=DEVICE)
except Exception as e:
    raise RuntimeError(f"Error cargando CLIP: {e}")

class CLIPEmbeddings:
    def __init__(self):
        self.model  = CLIP_VISUAL
        self.preproc = CLIP_PREPROC

    def embed_image(self, img: Image.Image):
        with torch.no_grad():
            vec = self.model.encode_image(
                self.preproc(img).unsqueeze(0).to(DEVICE)
            )
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec[0]

CLIP_EMB = CLIPEmbeddings()

# 3.  Ollama – captions y embeddings de texto ────────────────
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage

def caption_image_llava(image: Image.Image) -> str:
    """Envía la imagen a LLaVA como data-URL base64 y devuelve el caption."""
    # Codificar la imagen en memoria
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    chat = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_VISION_MODEL)
    msg  = HumanMessage(content=[
        {"type": "image_url", "image_url": data_url},
        {"type": "text",      "text": "Describe the image briefly."}
    ])
    return chat.invoke([msg]).content.strip()

TEXT_EMB = OllamaEmbeddings(model=OLLAMA_TEXT_EMB_MODEL,
                            base_url=OLLAMA_BASE_URL)

# 4.  Extraer frames ─────────────────────────────────────────
def extract_frames(video_path: str, every_n_seconds: int = 2) -> List[Image.Image]:
    clip_v   = VideoFileClip(video_path)
    duration = int(clip_v.duration)
    frames   = []
    for t in range(0, duration, every_n_seconds):
        frame = clip_v.get_frame(t)          # numpy array (H, W, 3)
        frames.append(Image.fromarray(frame))
    clip_v.close()
    return frames

# 5.  Pipeline completo ─────────────────────────────────────
def process_video(video_path: str):
    frames = extract_frames(video_path, EVERY_N_SECONDS)
    if not frames:
        raise RuntimeError("No se extrajeron frames.")

    captions  = []
    clip_vecs = []

    for idx, frame in enumerate(frames):
        print(f"Frame {idx+1}/{len(frames)}...")
        caption = caption_image_llava(frame)
        print("  Caption:", caption)
        captions.append(caption)
        clip_vecs.append(CLIP_EMB.embed_image(frame))

    # Embedding global del texto (con Ollama)
    joined    = " ".join(captions)
    sent_vec  = torch.tensor(TEXT_EMB.embed_query(joined))

    # Promedio de embeddings CLIP
    clip_avg  = torch.stack(clip_vecs).mean(dim=0)

    return captions, sent_vec, clip_avg

# 6.  Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python videosemb.py <video>")
        sys.exit(1)

    video_file = sys.argv[1]
    if not Path(video_file).exists():
        print("Archivo no encontrado:", video_file)
        sys.exit(1)

    caps, sent_emb, clip_emb = process_video(video_file)

    print("\n── Resultado ──────────────────────────")
    print("Captions:", caps)
    print(f"Sentence embedding (dim={len(sent_emb)}):", sent_emb[:10], "...")
    print(f"CLIP embedding (dim={len(clip_emb)}):", clip_emb[:10], "...")

    # Ejemplo para guardar resultados
    # torch.save(clip_emb.cpu(), "clip_avg.pt")
    # with open("sent_emb.json", "w") as f:
    #     import json; json.dump(sent_emb.tolist(), f, indent=2)
