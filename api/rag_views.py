# rag/views.py
from collections import defaultdict
from typing import List, Dict, Tuple
from PIL import Image

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, JSONParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import weaviate.classes as wvc
from .weaviate_client import CLIENT 
from .resources import (
    CLIP_EMB, TEXT_EMB, CLASSES, IMAGE_VECTOR_FIELDS, short_query
)

def run_wv_query(
    cls: str,
    vec: List[float],
    k: int,
    vfield: str | None = None,
    props: Tuple[str, ...] = ("file_location", "doc_id"),
) -> List[Dict]:
    """
    Ejecuta near-vector con SDK v4 (sin encadenar métodos extra).
    """
    coll = CLIENT.collections.get(cls)

    res = coll.query.near_vector(
        vec,                       # vector posicional
        target_vector=vfield,      # None ⇒ vector por defecto
        limit=k,                   # cuántos
       return_properties=list(props),
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )  # ← ya ejecuta y devuelve QueryReturn

    hits = res.objects                # lista DataObject
    out = []
    for h in hits:
        item = {
            "id": h.uuid,
            "distance": h.metadata.distance,
        }
        item.update({p: h.properties.get(p, "") for p in props})
        out.append(item)
    return out




class MultiModalSearchView(APIView):
    parser_classes = (MultiPartParser, JSONParser, FormParser)

    def post(self, request, *args, **kwargs):
        data = request.data
        query_txt = data.get("query", "")

        flags = {
            "text": data.get("search_text") in ("true", True, "1"),
            "pdf": data.get("search_pdf") in ("true", True, "1"),
            "image": data.get("search_image") in ("true", True, "1"),
        }
        ks = {
            "text": int(data.get("k_text", 5)),
            "pdf": int(data.get("k_pdf", 20)),
            "image": int(data.get("k_image", 10)),
        }
        if not any(flags.values()):
            return Response(
                {"error": "Selecciona al menos un dominio de búsqueda."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        results = defaultdict(list)

        # ── TEXT / PDF ─────────────────────────────────────
        if flags["text"] or flags["pdf"]:
            vec_txt = TEXT_EMB.embed_query(short_query(query_txt))
            if flags["text"]:
                results["text_results"] = run_wv_query(
                    CLASSES["text"], vec_txt, ks["text"]
                )
            if flags["pdf"]:
                results["pdf_results"] = run_wv_query(
                    CLASSES["pdf"],
                    vec_txt,
                    ks["pdf"],
                    props=("original_doc_id", "page_number", "chunk_sequence",   "text_chunk", ),
                )

        # ── IMAGES ─────────────────────────────────────────
        if flags["image"]:
            use_clip = data.get("use_clip", "true") in ("true", True, "1")
            use_ocr = data.get("use_ocr") in ("true", True, "1")
            use_des = data.get("use_description") in ("true", True, "1")

            if "file" in request.FILES:  # imagen→imagen
                img = Image.open(request.FILES["file"]).convert("RGB")
                vec_clip = CLIP_EMB.embed_image_pil(img)
                results["image_clip"] = run_wv_query(
                    CLASSES["image"], vec_clip, ks["image"], IMAGE_VECTOR_FIELDS["clip"]
                )
            else:  # texto→imagen
                qvec_clip = CLIP_EMB.embed_text(short_query(query_txt))
                if use_clip:
                    results["image_clip"] = run_wv_query(
                        CLASSES["image"], qvec_clip, ks["image"], IMAGE_VECTOR_FIELDS["clip"]
                    )
                if use_ocr:
                    vec = TEXT_EMB.embed_query(short_query(query_txt))
                    results["image_ocr"] = run_wv_query(
                        CLASSES["image"], vec, ks["image"], IMAGE_VECTOR_FIELDS["ocr"]
                    )
                if use_des:
                    vec = TEXT_EMB.embed_query(short_query(query_txt))
                    results["image_description"] = run_wv_query(
                        CLASSES["image"], vec, ks["image"], IMAGE_VECTOR_FIELDS["des"]
                    )

        return Response({"query_used": query_txt, **results})
