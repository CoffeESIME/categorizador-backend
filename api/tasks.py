import os
from typing import List, Dict, Any
from celery import shared_task
from django.conf import settings
from langchain_ollama import OllamaEmbeddings

from .services.processing import (
    _encode_image,
    _encode_video,
    _encode_audio,
    _transcribe_audio,
    process_text_embeddings
)
from .pdf_embeddings import embed_pdf_and_store
from .embeddings_to_neo import (
    limpiar_meta,
    store_embedding,
    guardar_imagen_en_weaviate,
    guardar_video_en_weaviate,
    guardar_audio_en_weaviate,
)
from .weaviate_client import CLIENT

@shared_task(bind=True)
def process_pdf_task(self, file_id: str, meta: Dict[str, Any]) -> None:
    pdf_path = os.path.join("uploads", meta["file_location"])
    embed_pdf_and_store(
        pdf_path=pdf_path, original_doc_id=file_id, client=CLIENT
    )
    store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)

@shared_task(bind=True)
def process_image_with_description_task(self, file_id: str, meta: Dict[str, Any], image_path: str) -> None:
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

@shared_task(bind=True)
def process_ocr_with_image_task(self, file_id: str, meta: Dict[str, Any], image_path: str) -> None:
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

@shared_task(bind=True)
def process_text_embeddings_task(self, meta: Dict[str, Any]) -> List[float]:
    return process_text_embeddings(meta)

@shared_task(bind=True)
def process_audio_file_task(
    self,
    file_id: str,
    meta: Dict[str, Any],
    audio_path: str,
    transcribe: bool = False,
) -> None:
    vec_audio = _encode_audio(audio_path)

    if transcribe:
        text = _transcribe_audio(audio_path)
        meta["content"] = text

    vec_text = process_text_embeddings(meta)

    propiedades = limpiar_meta(meta)
    guardar_audio_en_weaviate(
        client=CLIENT,
        meta=propiedades,
        vec_audio=vec_audio,
        vec_text=vec_text,
    )
    store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)

@shared_task(bind=True)
def process_video_file_task(
    self,
    file_id: str,
    meta: Dict[str, Any],
    video_path: str,
    include_audio: bool = False,
    transcribe_audio: bool = False,
) -> None:
    vec_video = _encode_video(video_path)
    vec_audio = None
    if include_audio or transcribe_audio:
        vec_audio = _encode_audio(video_path)
    if transcribe_audio:
        text = _transcribe_audio(video_path)
        meta["content"] = text
    
    vec_text = process_text_embeddings(meta)

    propiedades = limpiar_meta(meta)
    guardar_video_en_weaviate(
        client=CLIENT,
        meta=propiedades,
        vec_video=vec_video,
        vec_audio=vec_audio,
        vec_text=vec_text,
    )
    store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)
