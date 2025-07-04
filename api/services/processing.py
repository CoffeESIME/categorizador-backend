import os
from typing import List, Dict, Any

from django.conf import settings
from PIL import Image
import torch
import clip
from langchain_ollama import OllamaEmbeddings

from ..pdf_embeddings import embed_pdf_and_store
from ..embeddings_to_neo import (
    limpiar_meta,
    store_embedding,
    guardar_imagen_en_weaviate,
)
from ..weaviate_client import CLIENT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    MODEL_CLIP, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
except Exception:  # pragma: no cover - handled during runtime
    MODEL_CLIP, PREPROCESS = None, None


def process_pdf(file_id: str, meta: Dict[str, Any]) -> None:
    pdf_path = os.path.join("uploads", meta["file_location"])
    embeddings_created, uuids = embed_pdf_and_store(
        pdf_path=pdf_path, original_doc_id=file_id, client=CLIENT
    )
    store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)


def _encode_image(image_path: str) -> List[float]:
    if not MODEL_CLIP or not PREPROCESS:
        raise RuntimeError("CLIP model not available")
    image = Image.open(image_path).convert("RGB")
    image_input = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = MODEL_CLIP.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].tolist()


def process_image_with_description(file_id: str, meta: Dict[str, Any], image_path: str) -> None:
    embedding_clip = _encode_image(image_path)
    texts_for_embedding: List[str] = []
    if meta.get("style"):
        texts_for_embedding.append(meta["style"])
    if meta.get("composition"):
        texts_for_embedding.append(meta["composition"])
    if meta.get("analysis"):
        texts_for_embedding.append(meta["analysis"])
    if meta.get("description"):
        texts_for_embedding.append(meta["description"])
    embedding_des = []
    if texts_for_embedding:
        embedding_model = OllamaEmbeddings(
            model=settings.DEFAULT_EMBED_MODEL, base_url=settings.LLM_BASE_URL
        )
        embedding_des = embedding_model.embed_documents([" ".join(texts_for_embedding)])[0]

    propiedades = limpiar_meta(meta)
    guardar_imagen_en_weaviate(
        client=CLIENT,
        meta=propiedades,
        vec_clip=embedding_clip,
        vec_desc=embedding_des,
    )
    store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)


def process_ocr_with_image(file_id: str, meta: Dict[str, Any], image_path: str) -> None:
    ocr_text = meta.get("ocr_text", "")
    if not ocr_text:
        raise ValueError("No se encontró 'ocr_text' para procesamiento OCR")
    meta["content"] = ocr_text
    embedding_model = OllamaEmbeddings(
        model=settings.DEFAULT_EMBED_MODEL, base_url=settings.LLM_BASE_URL
    )
    embedding_text = embedding_model.embed_documents([ocr_text])[0]

    embedding_clip = _encode_image(image_path)
    propiedades = limpiar_meta(meta)
    guardar_imagen_en_weaviate(
        client=CLIENT,
        meta=propiedades,
        vec_clip=embedding_clip,
        vec_ocr=embedding_text,
    )
    store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)


def process_text_embeddings(meta: Dict[str, Any]) -> List[float]:
    texts_for_embedding: List[str] = []
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
    if not combined_text:
        return []

    embedding_model = OllamaEmbeddings(
        model=settings.DEFAULT_EMBED_MODEL, base_url=settings.LLM_BASE_URL
    )
    return embedding_model.embed_documents([combined_text])[0]
