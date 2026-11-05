import os
import time
import cv2
import numpy as np
from dotenv import load_dotenv
from roommate_recognizer.recognition.engine import FaceEngine
from roommate_recognizer.sources.rtsp import open_rtsp

PROCESS_EVERY = 5

load_dotenv()
gallery = np.load("data/enrolled/gallery.npz")
engine = FaceEngine()
capture = open_rtsp(os.environ["RTSP_URL"])

frame_number = 0
detections = []
milliseconds = 0

while True:
    ok, frame = capture.read()
    if not ok:
        print("Frame read failed")
        break

    if frame_number % PROCESS_EVERY == 0:
        start = time.perf_counter()
        result = engine.person_detector(frame, classes=[0], conf=0.5, device=engine.device, verbose=False)[0]
        detections = []
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = [int(value) for value in box]
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            label = "no face"
            if crop.size:
                faces = engine.embed_faces(crop)
                if faces:
                    name, score = engine.identify(faces[0], gallery["names"], gallery["embeddings"])
                    label = f"{name} {score:.2f}"
            detections.append((x1, y1, x2, y2, label))
        milliseconds = (time.perf_counter() - start) * 1000

    for x1, y1, x2, y2, label in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"{milliseconds:.0f} ms", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Live recognition", frame)
    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()