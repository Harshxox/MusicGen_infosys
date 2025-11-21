# backend/gtzan.py
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

YAMNET_SR = 16000

print("Loading YAMNet (TF Hub) and classifier model (this may take a second)...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
clf = load_model("models/yamnet_gtzan_final.h5")

# load genres from the embeddings file names (quick hack)
import numpy as np
meta = np.load("../data/gtzan_embeddings.npz", allow_pickle=True)
GENRES = meta['genres'].tolist()

def extract_yamnet_features(file_path):
    wav, sr = librosa.load(file_path, sr=YAMNET_SR, mono=True)
    wav = wav.astype('float32')
    scores, embeddings, spec = yamnet(tf.convert_to_tensor(wav))
    return embeddings.numpy().mean(axis=0)

def predict_genre_from_file(file_path):
    feat = extract_yamnet_features(file_path).reshape(1, -1)
    probs = clf.predict(feat)[0]
    idx = int(np.argmax(probs))
    return {
        "genre": str(GENRES[idx]),
        "confidence": float(probs[idx]),
        "probs": probs.tolist()
    }

# quick test if run directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gtzan.py <audiofile>")
        sys.exit(1)
    res = predict_genre_from_file(sys.argv[1])
    print(res)
