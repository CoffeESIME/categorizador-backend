import torchaudio, torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ---- parámetros ----
MODEL_ID  = "facebook/wav2vec2-xls-r-1b"
AUDIO_IN  = "test.mp3"               # mp3, ogg, flac…
DEVICE    = "cuda"                      # o "cpu"

# ---- carga modelo ----
fe  = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
enc = Wav2Vec2Model.from_pretrained(MODEL_ID).to(DEVICE).eval()

# ---- lee audio con torchaudio.load ----
wav, sr = torchaudio.load(AUDIO_IN)     # decodifica con FFmpeg
wav = wav.mean(dim=0, keepdim=True)     # mono
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)

# ---- extrae embedding ----
inputs = fe(wav.squeeze().numpy(), sampling_rate=16000,
            return_tensors="pt").input_values.to(DEVICE)
with torch.no_grad():
    vec = enc(inputs).last_hidden_state.mean(dim=1).squeeze()  # (1024,)
print(vec.shape)