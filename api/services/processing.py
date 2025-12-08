import os
import logging
from typing import List, Dict, Any
import torch
from PIL import Image
from langchain_ollama import OllamaEmbeddings
from django.conf import settings

from .model_loader import model_loader

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _encode_image(image_path: str) -> List[float]:
    logger.debug(f"Encoding image: {image_path}")
    try:
        model, preprocess = model_loader.get_clip_model()
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(model_loader.device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}", exc_info=True)
        raise e

def _encode_video(video_path: str) -> List[float]:
    logger.debug(f"Encoding video: {video_path}")
    try:
        import decord
        
        processor = model_loader.get_xclip_processor()
        model = model_loader.get_xclip_model()
        device = model_loader.device

        # ── VideoReader SIN context manager ───────────────
        vr = decord.VideoReader(video_path, width=224, height=224)
        try:
            idx = list(range(0, len(vr), max(1, len(vr) // 8)))[:8]
            frames = vr.get_batch(idx).asnumpy()          # ndarray (N, H, W, C)
        finally:
            del vr                                        # libera el handle nativo

        inputs = processor(videos=list(frames), return_tensors="pt").to(device)
        with torch.no_grad():
            video_embeds = model.get_video_features(**inputs)
            vec = video_embeds[0] / video_embeds[0].norm()

        return vec.cpu().numpy().tolist()
    except Exception as e:
        logger.error(f"Error encoding video {video_path}: {e}", exc_info=True)
        raise e

def _encode_audio(audio_path: str) -> List[float]:
    logger.debug(f"Encoding audio: {audio_path}")
    try:
        try:
            import torchaudio
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        except Exception as e:
            raise RuntimeError(f"Audio embedding deps missing: {e}")

        # Note: Wav2Vec2 is not in singleton yet, loading locally for now
        # We could add it to singleton but keeping it here for now to minimize changes
        
        device = model_loader.device
        fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-1b")
        model = (
            Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-1b")
            .to(device)
            .eval()
        )

        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        inputs = fe(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            vec = model(inputs).last_hidden_state.mean(dim=1).squeeze()
        return vec.cpu().numpy().tolist()
    except Exception as e:
        logger.error(f"Error encoding audio {audio_path}: {e}", exc_info=True)
        raise e

def _transcribe_audio(audio_path: str) -> str:
    logger.debug(f"Transcribing audio: {audio_path}")
    try:
        model = model_loader.get_whisper_model()
        segments, _ = model.transcribe(audio_path, language="es")
        return " ".join(s.text for s in segments)
    except Exception as e:
        logger.error(f"Error transcribing audio {audio_path}: {e}", exc_info=True)
        raise e

def process_text_embeddings(meta: Dict[str, Any]) -> List[float]:
    logger.debug("Processing text embeddings")
    try:
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
            logger.warning("No text content found for embedding")
            return []

        embedding_model = OllamaEmbeddings(
            model=settings.DEFAULT_EMBED_MODEL, base_url=settings.LLM_BASE_URL
        )
        return embedding_model.embed_documents([combined_text])[0]
    except Exception as e:
        logger.error(f"Error processing text embeddings: {e}", exc_info=True)
        raise e
