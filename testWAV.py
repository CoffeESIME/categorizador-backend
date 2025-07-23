from faster_whisper import WhisperModel
model = WhisperModel("medium",           # o "small"
                     device="cpu",
                     compute_type="int8")  # <- NO float16
segments, info = model.transcribe("test.mp3", language="es")
print(" ".join(s.text for s in segments))
