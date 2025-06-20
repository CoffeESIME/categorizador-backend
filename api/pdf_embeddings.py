# utils/pdf_embeddings.py
import os
from typing import List, Dict, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from django.conf import settings

from .embeddings_to_neo import store_chunk_in_weaviate


def embed_pdf_and_store(
    *,                       # obligatoriamente por nombre ➜ mayor claridad
    pdf_path: str,
    original_doc_id: str,
    client,                  # instancia global de weaviate
) -> Tuple[int, List[str]]:
    """
    Devuelve (n_chunks_insertados, uuids_weaviate).
    Levanta excepción si el PDF no existe o está vacío.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No se encontró el PDF en {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()                      # 1 por página
    if not docs:
        raise ValueError("PDF vacío o sin texto extraíble")

    splitter_parent = RecursiveCharacterTextSplitter(
        chunk_size=2400, chunk_overlap=200
    )
    splitter_child = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    )
    # 1) partimos páginas grandes en “pads” (parent) …
    parents = splitter_parent.split_documents(docs)
    # 2) partimos cada parent en “child” (lo que indexaremos)
    children = []
    for p in parents:
        children.extend(splitter_child.split_documents([p]))

    emb_model = OllamaEmbeddings(model=settings.DEFAULT_EMBED_MODEL,
                                 base_url=settings.LLM_BASE_URL)
    texts = [c.page_content for c in children]
    vectors = emb_model.embed_documents(texts)

    uuids = []
    for idx, (chunk, vec) in enumerate(zip(children, vectors)):
        chunk_meta = {
            "chunk_sequence": idx,
            "page_number": chunk.metadata.get("page", -1),
        }
        uuid = store_chunk_in_weaviate(
            client=client,
            chunk_text=chunk.page_content,
            embedding=vec,
            original_doc_id=original_doc_id,
            chunk_metadata=chunk_meta,
        )
        if uuid:
            uuids.append(uuid)

    return len(uuids), uuids
