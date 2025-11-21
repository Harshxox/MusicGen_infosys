# backend/train_classifier.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

DATA_FILE = "../data/gtzan_embeddings.npz"
OUT_MODEL = "models/yamnet_gtzan_final.h5"
os.makedirs("models", exist_ok=True)

data = np.load(DATA_FILE, allow_pickle=True)
X = data['X']
y = data['y']
genres = data['genres']

num_classes = len(genres)
y_cat = to_categorical(y, num_classes=num_classes)

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.25, random_state=42, stratify=y)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

model.save(OUT_MODEL)
print("Saved model to", OUT_MODEL)
print("Training complete.")