from ultralytics import YOLO
import cv2
import torch
from PIL import Image
import numpy as np


class FrameClassifier:
    def __init__(self, frame_model_path='models/frame.pt', codes_model_path='models/codes.pt'):

        self.frame_model = YOLO(frame_model_path)
        self.codes_model = YOLO(codes_model_path)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")


    def _crop_to_frame(self, image_path, confidence=0.3):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        original_img = img.copy()
        h, w = img.shape[:2]

        results = self.frame_model.predict(source=img, conf=confidence, device=self.device)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print("Рамы не обнаружены, используется исходное изображение")
            return Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

        best_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = map(int, best_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped_img = original_img[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))


    def classify_frame(self, image_path, confidence=0.01):
        processed_img = self._crop_to_frame(image_path)
        img_array = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)

        results = self.codes_model.predict(
            source=img_array,
            device=self.device,
            conf=confidence
        )

        if len(results[0].boxes) == 0:
            return None

        best_idx = np.argmax(results[0].boxes.conf.cpu().numpy())
        best_class = results[0].boxes.cls.cpu().numpy().astype(int)[best_idx]
        best_conf = results[0].boxes.conf.cpu().numpy()[best_idx]

        print(f"Найден производитель: {self.codes_model.names[best_class]} (уверенность: {best_conf:.2f})")
        return self.codes_model.names[best_class]