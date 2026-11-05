import os
import time
import numpy as np
from dotenv import load_dotenv
from roommate_recognizer.recognition.engine import FaceEngine
from roommate_recognizer.sources.rtsp import open_rtsp

load_dotenv()
gallery = np.load("data/enrolled/gallery.npz")
engine = FaceEngine()
capture = open_rtsp(os.environ["RTSP_URL"])

while True:
    ok, frame = capture.read()
    if not ok:
        break
    start = time.perf_counter()
    embeddings = engine.embed_people(frame)
    milliseconds = (time.perf_counter() - start) * 1000
    for embedding in embeddings:
        name, score = engine.identify(embedding, gallery["names"], gallery["embeddings"])
        print(f"{name} score={score:.2f} time={milliseconds:.0f}ms")