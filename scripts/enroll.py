from pathlib import Path
import cv2
import numpy as np
from roommate_recognizer.recognition.engine import FaceEngine

engine = FaceEngine()
names, embeddings = [], []

for person in Path("data/enrolled").iterdir():
    if not person.is_dir():
        continue
    for photo in person.iterdir():
        image = cv2.imread(str(photo))
        if image is None:
            print(f"Skipped unreadable file: {photo}")
            continue
        faces = engine.embed_faces(image)
        if len(faces) != 1:
            print(f"Skipped {photo}: found {len(faces)} faces")
            continue
        names.append(person.name)
        embeddings.append(faces[0])

np.savez("data/enrolled/gallery.npz", names=np.array(names), embeddings=np.array(embeddings))
print(f"Enrolled {len(names)} photos")