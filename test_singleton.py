from api.services.model_loader import model_loader
import torch

def test_singleton():
    print("Testing ModelSingleton...")
    
    # Test 1: Check initial state (should be None)
    assert model_loader._xclip_model is None
    assert model_loader._whisper_model is None
    print("Initial state: OK (Models not loaded)")

    # Test 2: Load XCLIP
    print("Requesting XCLIP model...")
    xclip = model_loader.get_xclip_model()
    assert xclip is not None
    assert model_loader._xclip_model is not None
    print(f"XCLIP loaded on: {xclip.device}")
    
    # Test 3: Singleton behavior (same instance)
    xclip2 = model_loader.get_xclip_model()
    assert xclip is xclip2
    print("Singleton behavior: OK")

    # Test 4: Load Whisper
    print("Requesting Whisper model...")
    whisper = model_loader.get_whisper_model()
    assert whisper is not None
    print("Whisper loaded")

    # Test 5: Check GPU usage if available
    if torch.cuda.is_available():
        assert model_loader.device == "cuda"
        print("GPU usage: OK")
    else:
        print("GPU not available, running on CPU")

if __name__ == "__main__":
    test_singleton()
