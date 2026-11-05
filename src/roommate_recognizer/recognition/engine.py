import cv2
import numpy as np
import torch
from ultralytics import YOLO


class FaceEngine:
    def __init__(self, threshold=0.363):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.person_detector = YOLO("yolov8n.pt")
        self.detector = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (320, 320), 0.8)
        self.recognizer = cv2.FaceRecognizerSF.create("models/sface.onnx", "")
        self.threshold = threshold

    def embed_faces(self, image):
        height, width = image.shape[:2]
        scale = min(1.0, 1280 / max(height, width))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        embeddings = []
        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            embedding = self.recognizer.feature(aligned).flatten()
            embeddings.append(embedding / np.linalg.norm(embedding))
        return embeddings

    def embed_people(self, image):
        result = self.person_detector(image, classes=[0], conf=0.5, device=self.device, verbose=False)[0]
        embeddings = []
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = [int(value) for value in box]
            crop = image[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size:
                embeddings.extend(self.embed_faces(crop))
        return embeddings

    def identify(self, embedding, names, gallery):
        scores = gallery @ embedding
        best = int(np.argmax(scores))
        score = float(scores[best])
        name = str(names[best]) if score >= self.threshold else "unknown"
        return name, score