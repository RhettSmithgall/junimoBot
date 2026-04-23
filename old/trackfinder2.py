import cv2
import numpy as np
import os
import mss
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train22/weights/best.pt"

OUTPUT_IMG_FOLDER = "trackTraining/train/images"
OUTPUT_LABEL_FOLDER = "trackTraining/train/labels"

NUM_SLICES = 5
THRESHOLD = 0

model = YOLO(MODEL_PATH)


def ensure_dirs():
    os.makedirs(OUTPUT_IMG_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)


def save_yolo_labels(path, detections, img_w, img_h):
    with open(path, "w") as f:
        for cls, x1, y1, x2, y2 in detections:

            w = x2 - x1
            h = y2 - y1

            x_center = (x1 + w / 2) / img_w
            y_center = (y1 + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            f.write(f"{cls} {x_center} {y_center} {w_norm} {h_norm}\n")


def process():
    ensure_dirs()

    monitor = {"top": 150, "left": 10, "width": 1500, "height": 825}

    save_index = 22

    with mss.mss() as sct:

        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            h, w, _ = frame.shape
            slice_width = w // NUM_SLICES

            annotated = frame.copy()
            detections = []

            # ----------------------------
            # RUN SLICE INFERENCE
            # ----------------------------
            for i in range(NUM_SLICES):
                x_start = i * slice_width
                x_end = (i + 1) * slice_width

                slice_img = frame[:, x_start:x_end]

                results = model(slice_img, verbose=False)[0]

                if results.boxes is None:
                    continue

                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if conf < THRESHOLD:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # convert to full frame coords
                    gx1 = int(x1 + x_start)
                    gy1 = int(y1)
                    gx2 = int(x2 + x_start)
                    gy2 = int(y2)

                    detections.append((cls, gx1, gy1, gx2, gy2))

            # ----------------------------
            # DRAW ON FULL FRAME
            # ----------------------------
            for cls, x1, y1, x2, y2 in detections:

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"{cls}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            # ----------------------------
            # SHOW FRAME
            # ----------------------------
            cv2.imshow("Full Detection", annotated)

            key = cv2.waitKey(0)

            # ----------------------------
            # SAVE ON 'Y'
            # ----------------------------
            if key == ord('y') and len(detections) > 0:

                img_name = f"{save_index}.png"
                label_name = f"{save_index}.txt"

                img_path = os.path.join(OUTPUT_IMG_FOLDER, img_name)
                label_path = os.path.join(OUTPUT_LABEL_FOLDER, label_name)

                cv2.imwrite(img_path, annotated)

                save_yolo_labels(
                    label_path,
                    detections,
                    w,
                    h
                )

                print(f"[SAVED] {img_name} ({len(detections)} objects)")
                save_index += 1

            elif key == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    process()