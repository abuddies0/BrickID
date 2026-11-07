import numpy as np
import onnxruntime as ort


class BrickDetector:
    INPUT_SIZE = 512
    STRIDES = (8, 16, 32)
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7

    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def decode_boxes(raw_boxes):
        """Decode YOLOv8 distribution-based box predictions."""

        # [N, 64] -> [N, 4, 16]
        distances = raw_boxes.reshape(-1, 4, 16)

        # Softmax over the 16 possible distance values.
        distances = distances - np.max(
            distances, axis=-1, keepdims=True
        )

        probabilities = np.exp(distances)
        probabilities /= probabilities.sum(
            axis=-1, keepdims=True
        )

        # Expected distance for each of the four coordinates.
        bins = np.arange(16, dtype=np.float32)
        return np.sum(probabilities * bins, axis=-1)

    @staticmethod
    def generate_anchors(size):
        """Generate YOLOv8 anchor centers and stride values."""

        anchors = []
        strides = []

        for stride in (8, 16, 32):
            height = size // stride
            width = size // stride

            y, x = np.meshgrid(
                np.arange(height),
                np.arange(width),
                indexing="ij",
            )

            points = np.stack(
                [x.reshape(-1), y.reshape(-1)],
                axis=-1,
            ).astype(np.float32)

            anchors.append(points + 0.5)
            strides.append(
                np.full(len(points), stride, dtype=np.float32)
            )

        return np.concatenate(anchors), np.concatenate(strides)

    @staticmethod
    def nms(boxes, scores, threshold):
        """Greedy non-maximum suppression."""

        order = np.argsort(scores)[::-1]
        keep = []

        while len(order):
            current = order[0]
            keep.append(current)

            if len(order) == 1:
                break

            remaining = order[1:]

            x1 = np.maximum(
                boxes[current, 0], boxes[remaining, 0]
            )
            y1 = np.maximum(
                boxes[current, 1], boxes[remaining, 1]
            )
            x2 = np.minimum(
                boxes[current, 2], boxes[remaining, 2]
            )
            y2 = np.minimum(
                boxes[current, 3], boxes[remaining, 3]
            )

            intersection = (
                np.maximum(0, x2 - x1)
                * np.maximum(0, y2 - y1)
            )

            area_current = (
                (boxes[current, 2] - boxes[current, 0])
                * (boxes[current, 3] - boxes[current, 1])
            )

            area_remaining = (
                (boxes[remaining, 2] - boxes[remaining, 0])
                * (boxes[remaining, 3] - boxes[remaining, 1])
            )

            union = area_current + area_remaining - intersection
            iou = intersection / np.maximum(union, 1e-7)

            order = remaining[iou <= threshold]

        return np.asarray(keep, dtype=np.int64)

    def predict(self, image):
        """
        Predict bricks in a preprocessed image.

        image: NumPy array [512, 512, 3], RGB, float32, 0-255.

        Returns: [N, 5] array containing
                 x1, y1, x2, y2, confidence.
        """

        image = np.asarray(image, dtype=np.float32)

        if image.shape != (512, 512, 3):
            raise ValueError("Expected a 512x512 RGB image.")

        image = image[np.newaxis, ...]

        raw_outputs = self.session.run(
            None,
            {self.input_name: image},
        )

        output_names = [
            output.name for output in self.session.get_outputs()
        ]

        predictions = dict(zip(output_names, raw_outputs))

        raw_boxes = predictions["boxes"][0]
        scores = predictions["classes"][0, :, 0]

        # Decode distances from the 64 regression channels.
        distances = self.decode_boxes(raw_boxes)

        # Generate anchor points in grid coordinates.
        anchors, strides = self.generate_anchors(512)

        # Convert left/top/right/bottom distances to xyxy.
        left_top = anchors - distances[:, :2]
        right_bottom = anchors + distances[:, 2:]

        boxes = np.concatenate(
            [left_top, right_bottom],
            axis=-1,
        )

        # Convert grid coordinates to image pixels.
        boxes *= strides[:, np.newaxis]

        # Filter low-confidence predictions.
        valid = scores >= self.CONFIDENCE_THRESHOLD

        boxes = boxes[valid]
        scores = scores[valid]

        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        # Clip coordinates to image boundaries.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 512)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 512)

        # Remove invalid boxes.
        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        scores = scores[valid]

        # Suppress overlapping detections.
        keep = self.nms(
            boxes,
            scores,
            self.IOU_THRESHOLD,
        )

        return np.column_stack([
            boxes[keep],
            scores[keep],
        ]).astype(np.float32)



########
# TEST #
########
from PIL import Image, ImageDraw
from brick_detector import BrickDetector

detector = BrickDetector("model/brick_detector.onnx")

image = Image.open("person_holding_brick.png").convert("RGB")
image = image.resize((512, 512))

predictions = detector.predict(image)

draw = ImageDraw.Draw(image)

for x1, y1, x2, y2, confidence in predictions:
    draw.rectangle(
        (x1, y1, x2, y2),
        outline="lime",
        width=3,
    )

    draw.text(
        (x1, max(0, y1 - 15)),
        f"Brick: {confidence:.1%}",
        fill="lime",
    )

image.save("brick_detection_result.jpg")
image.show()

print(f"Detected {len(predictions)} bricks")