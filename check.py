from transformers import AutoProcessor, XCLIPModel
import decord, torch
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# Video → 8 frames (H,W,C) uint8
vr   = decord.VideoReader("sample2.mp4", width=224, height=224)
idx  = list(range(0, len(vr), max(1, len(vr)//8)))[:8]
frames = list(vr.get_batch(idx).asnumpy().astype("uint8"))

proc  = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch16").to(device).eval()

inputs = proc(
    text=["a man walking on a bridge at night"],
    videos=[frames],                # lista-de-vídeos → lista-de-frames
    return_tensors="pt"
).to(device)

with torch.no_grad():
    out = model(**inputs)
    sim = torch.nn.functional.cosine_similarity(
              out.video_embeds, out.text_embeds
          )[0]


sim = F.cosine_similarity(
        out.video_embeds,          # shape = (batch, 512)
        out.text_embeds,
        dim=-1                     # ← eje de embedding
      )[0]                         # (batch,) → escalar
print(f"cosine similarity = {sim.item():.3f}")
