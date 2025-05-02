import  torch, clip
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# ── 1. CLIP  (usa el que ya tenías) ──────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
try:
    CLIP_VISUAL, CLIP_PREPROC = clip.load("ViT-B/32", device=DEVICE)
except Exception:
    CLIP_VISUAL = CLIP_PREPROC = None

class CLIPEmbeddings:
    def __init__(self):
        self.model, self.preproc = CLIP_VISUAL, CLIP_PREPROC
        if self.model is None:
            raise RuntimeError("CLIP no cargó")

    def embed_text(self, txt: str):
        import torch
        with torch.no_grad():
            tokens = clip.tokenize([txt]).to(DEVICE)   # ← usa clip.tokenize
            t = self.model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
        return t[0].cpu().tolist()

    def embed_image_pil(self, img):
        import torch
        with torch.no_grad():
            v = self.model.encode_image(
                self.preproc(img).unsqueeze(0).to(DEVICE)
            )
            v = v / v.norm(dim=-1, keepdim=True)
        return v[0].cpu().tolist()

CLIP_EMB = CLIPEmbeddings()

# ── 2. Embeddings y LLM de texto ─────────────────────────────
TEXT_EMB = OllamaEmbeddings(
    model="granite-embedding:latest", 
    base_url="http://localhost:11434", 
                   
)

LLM_REWRITER = (
    ChatPromptTemplate.from_template(
        "Extrae máx 15 palabras clave para búsqueda vectorial:\n{q}"
        "Devuelve EXACTAMENTE 10 palabras clave, separadas por comas, "
        "sin numeración ni texto extra:\n{q}"
    )
    | ChatOllama(model="gemma3:latest", temperature=0.2, max_tokens=6400,  base_url="http://localhost:11434",  options={"format": "text"}   )
    | StrOutputParser()
).invoke

MAX_CLIP_TOKENS = 75   # 77 menos el <EOS> y márgenes

def safe_for_clip(text: str) -> str:
    """Corta el texto para que no exceda 75 tokens aprox."""
    words = text.strip().split()
    if len(words) <= MAX_CLIP_TOKENS:
        return text
    return " ".join(words[:MAX_CLIP_TOKENS])

def short_query(q: str) -> str:
    try:
        rewritten = LLM_REWRITER({"q": q}).strip()
    except Exception:
        rewritten = q
    return safe_for_clip(rewritten)

CLASSES = dict(text="Textos", pdf="PdfChunks", image="Imagenes")
IMAGE_VECTOR_FIELDS = dict(clip="vector_clip", ocr="vector_ocr", des="vector_des")
