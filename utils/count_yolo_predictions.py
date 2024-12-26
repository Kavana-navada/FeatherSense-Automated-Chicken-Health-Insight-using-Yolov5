import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class count_YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        row, col, _ = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        detections = preds[0]
        boxes, confidences, classes = [], [], []
        class_counts = {label: 0 for label in self.labels}

        image_w, image_h = input_image.shape[:2]
        x_factor, y_factor = image_w / INPUT_WH_YOLO, image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.35:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.45:
                    cx, cy, w, h = row[0:4]
                    left, top = int((cx - 0.5 * w) * x_factor), int((cy - 0.5 * h) * y_factor)
                    width, height = int(w * x_factor), int(h * y_factor)
                    box = np.array([left, top, width, height])
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        boxes_np, confidences_np = np.array(boxes).tolist(), np.array(confidences).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.35, 0.45)

        for i in indices:
            x, y, w, h = boxes_np[i]
            class_id = classes[i]
            class_name = self.labels[class_id]
            colors = self.generate_colors(class_id)
            class_counts[class_name] += 1  # Update class count

            text = f'{class_name}'
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            cv2.rectangle(image, (x, y - 50), (x + w, y), colors, -1)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 1)
        return image, class_counts

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])
