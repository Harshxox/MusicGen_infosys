# backend/extract_embeddings.py
import os
import numpy as np
from pathlib import Path
import librosa
import tensorflow_hub as hub
import tensorflow as tf

# Configuration
GTZAN_DIR = Path("../data/gtzan")   # adjust if your path differs
OUT_FILE = "../data/gtzan_embeddings.npz"
YAMNET_SR = 16000

print("Loading YAMNet (TF Hub). This will download the model if not present...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_yamnet_features(file_path):
    # Loads audio, resamples to 16k, returns mean embedding (1024-d)
    waveform, sr = librosa.load(file_path, sr=YAMNET_SR, mono=True)
    waveform = waveform.astype('float32')
    scores, embeddings, spectrogram = yamnet(tf.convert_to_tensor(waveform))
    emb_np = embeddings.numpy()
    return emb_np.mean(axis=0)

# Collect genre directories
genres = sorted([p.name for p in GTZAN_DIR.iterdir() if p.is_dir()])
print("Detected genres:", genres)

X = []
y = []
filenames = []

for i, genre in enumerate(genres):
    folder = GTZAN_DIR / genre
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() not in ['.wav', '.au', '.mp3', '.ogg', '.flac']:
            continue
        try:
            feat = extract_yamnet_features(str(file))
            X.append(feat)
            y.append(i)
            filenames.append(str(file))
            print(f"Processed {file.name} -> genre {genre}")
        except Exception as e:
            print("Failed to process", file, "error:", e)

X = np.vstack(X)
y = np.array(y)
np.savez_compressed(OUT_FILE, X=X, y=y, genres=np.array(genres), filenames=np.array(filenames))
print("Saved embeddings to", OUT_FILE)
