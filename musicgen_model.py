
# musicgen_model.py
from transformers import pipeline
import scipy.io.wavfile as wavfile
import io

print("⏳ Loading MusicGen model (facebook/musicgen-small)... this may take a few minutes")
musicgen = pipeline("text-to-audio", model="facebook/musicgen-small",)
print("✅ Model loaded successfully!")

def generate_music(prompt: str, duration: float = 10.0):
    """Generate music from text and return a WAV audio buffer."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    print(f"🎵 Generating music for: {prompt}")
    result = musicgen(prompt, forward_params={"do_sample": True})
    audio = result["audio"]
    sr = result["sampling_rate"]

    buffer = io.BytesIO()
    wavfile.write(buffer, sr, audio)
    buffer.seek(0)
    return buffer
