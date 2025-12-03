import torch
from transformers import AutoProcessor, XCLIPModel
from faster_whisper import WhisperModel

class ModelSingleton:
    _instance = None
    _xclip_model = None
    _xclip_processor = None
    _whisper_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def get_xclip_model(self):
        if self._xclip_model is None:
            print(f"Loading XCLIP model on {self.device}...")
            self._xclip_model = (
                XCLIPModel.from_pretrained("microsoft/xclip-base-patch16")
                .to(self.device)
                .eval()
            )
        return self._xclip_model

    def get_xclip_processor(self):
        if self._xclip_processor is None:
            print("Loading XCLIP processor...")
            self._xclip_processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
        return self._xclip_processor

    def get_whisper_model(self, model_size="medium", compute_type="int8"):
        if self._whisper_model is None:
            print(f"Loading Whisper model ({model_size}) on {self.device}...")
            self._whisper_model = WhisperModel(
                model_size, 
                device=self.device, 
                compute_type=compute_type
            )
        return self._whisper_model

# Global instance
model_loader = ModelSingleton()
