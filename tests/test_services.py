import os
from unittest import mock

import pytest

from api.services import processing


def test_process_pdf_calls_dependencies(tmp_path):
    meta = {"file_location": "docs/sample.pdf"}
    with mock.patch.object(processing, "embed_pdf_and_store") as m_embed, \
         mock.patch.object(processing, "store_embedding") as m_store:
        m_embed.return_value = (1, ["uuid"])
        processing.process_pdf("file1", meta)
        m_embed.assert_called_once_with(
            pdf_path=os.path.join("uploads", "docs/sample.pdf"),
            original_doc_id="file1",
            client=processing.CLIENT,
        )
        m_store.assert_called_once()


def test_process_image_with_description_calls_services(tmp_path):
    meta = {"file_location": "img.jpg", "style": "s"}
    with mock.patch.object(processing, "_encode_image", return_value=[0.1]), \
         mock.patch.object(processing, "OllamaEmbeddings") as m_emb, \
         mock.patch.object(processing, "guardar_imagen_en_weaviate") as m_save, \
         mock.patch.object(processing, "store_embedding") as m_store:
        m_emb.return_value.embed_documents.return_value = [[0.2]]
        processing.process_image_with_description("fid", meta, "path")
        m_save.assert_called_once()
        m_store.assert_called_once()


def test_process_ocr_with_image_without_text():
    with pytest.raises(ValueError):
        processing.process_ocr_with_image("fid", {}, "img")


def test_process_text_embeddings_returns_vector():
    meta = {"content": "hola"}
    with mock.patch.object(processing, "OllamaEmbeddings") as m_emb:
        m_emb.return_value.embed_documents.return_value = [[0.3]]
        vec = processing.process_text_embeddings(meta)
        assert vec == [0.3]
        m_emb.assert_called_once()
