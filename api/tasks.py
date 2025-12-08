import os
import logging
from typing import List, Dict, Any
from celery import shared_task
from django.conf import settings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

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
from .utils import download_file_from_minio

@shared_task(bind=True)
def process_pdf_task(self, file_id: str, meta: Dict[str, Any]) -> None:
    logger.info(f"Task [process_pdf_task] started for file_id: {file_id}")
    try:
        object_name = meta["file_location"]
        # Download from MinIO to temp file
        with download_file_from_minio(settings.AWS_STORAGE_BUCKET_NAME, object_name) as pdf_path:
            embed_pdf_and_store(
                pdf_path=pdf_path, original_doc_id=file_id, client=CLIENT
            )
            store_embedding(doc_id=file_id, embedding=[], label="UnconnectedDoc", meta=meta)
            logger.info(f"Task [process_pdf_task] completed for file_id: {file_id}")
    except Exception as e:
        logger.error(f"Task [process_pdf_task] failed for file_id: {file_id}: {e}", exc_info=True)
        raise e

@shared_task(bind=True)
def process_image_with_description_task(self, file_id: str, meta: Dict[str, Any], file_key: str) -> None:
    logger.info(f"Task [process_image_with_description_task] started for file_id: {file_id}")
    try:
        with download_file_from_minio(settings.AWS_STORAGE_BUCKET_NAME, file_key) as image_path:
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
            logger.info(f"Task [process_image_with_description_task] completed for file_id: {file_id}")
    except Exception as e:
        logger.error(f"Task [process_image_with_description_task] failed for file_id: {file_id}: {e}", exc_info=True)
        raise e

@shared_task(bind=True)
def process_ocr_with_image_task(self, file_id: str, meta: Dict[str, Any], file_key: str) -> None:
    logger.info(f"Task [process_ocr_with_image_task] started for file_id: {file_id}")
    try:
        ocr_text = meta.get("ocr_text", "")
        if not ocr_text:
            raise ValueError("No se encontró 'ocr_text' para procesamiento OCR")
        
        with download_file_from_minio(settings.AWS_STORAGE_BUCKET_NAME, file_key) as image_path:
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
            logger.info(f"Task [process_ocr_with_image_task] completed for file_id: {file_id}")
    except Exception as e:
        logger.error(f"Task [process_ocr_with_image_task] failed for file_id: {file_id}: {e}", exc_info=True)
        raise e

@shared_task(bind=True)
def process_text_embeddings_task(self, meta: Dict[str, Any]) -> List[float]:
    logger.info(f"Task [process_text_embeddings_task] started.")
    try:
        result = process_text_embeddings(meta)
        logger.info(f"Task [process_text_embeddings_task] completed.")
        return result
    except Exception as e:
        logger.error(f"Task [process_text_embeddings_task] failed: {e}", exc_info=True)
        raise e

@shared_task(bind=True)
def process_audio_file_task(
    self,
    file_id: str,
    meta: Dict[str, Any],
    file_key: str,
    transcribe: bool = False,
) -> None:
    logger.info(f"Task [process_audio_file_task] started for file_id: {file_id}")
    try:
        with download_file_from_minio(settings.AWS_STORAGE_BUCKET_NAME, file_key) as audio_path:
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
            logger.info(f"Task [process_audio_file_task] completed for file_id: {file_id}")
    except Exception as e:
        logger.error(f"Task [process_audio_file_task] failed for file_id: {file_id}: {e}", exc_info=True)
        raise e

@shared_task(bind=True)
def process_video_file_task(
    self,
    file_id: str,
    meta: Dict[str, Any],
    file_key: str,
    include_audio: bool = False,
    transcribe_audio: bool = False,
) -> None:
    logger.info(f"Task [process_video_file_task] started for file_id: {file_id}")
    try:
        with download_file_from_minio(settings.AWS_STORAGE_BUCKET_NAME, file_key) as video_path:
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
            logger.info(f"Task [process_video_file_task] completed for file_id: {file_id}")
    except Exception as e:
        logger.error(f"Task [process_video_file_task] failed for file_id: {file_id}: {e}", exc_info=True)
        raise e
