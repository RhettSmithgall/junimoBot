import cv2
import numpy as np
import os
import mss

# SETTINGS
TEMPLATE_FOLDER = "tracks"
OUTPUT_IMG_FOLDER = "trackTraining\\train"
OUTPUT_LABEL_FOLDER = "trackTraining\\val"

NUM_SLICES = 4
METHOD = cv2.TM_CCOEFF_NORMED
THRESHOLD = 0.3

CLASS_ID = 0


def load_templates(folder):
    templates = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append((file, img))
    return templates


def ensure_dirs():
    os.makedirs(OUTPUT_IMG_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)


def non_max_suppression(detections, overlapThresh=0.3):
    if len(detections) == 0:
        return []

    boxes = []
    scores = []

    for det in detections:
        x, y, w, h = det["box"]
        boxes.append([x, y, x + w, y + h])
        scores.append(det["score"])

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=THRESHOLD,
        nms_threshold=overlapThresh
    )

    filtered = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered.append(detections[i])

    return filtered


def process():
    save_index = 61
    templates = load_templates(TEMPLATE_FOLDER)

    monitor = {"top": 350, "left": 10, "width": 1500, "height": 725}

    with mss.mss() as sct:
        ensure_dirs()

        while True:
            sct_img = sct.grab(monitor)

            frame = np.array(sct_img)[:, :, :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            slice_width = w // NUM_SLICES

            for i in range(NUM_SLICES):
                x_start = i * slice_width
                x_end = x_start + slice_width

                slice_img = gray[:, x_start:x_end]
                slice_color = frame[:, x_start:x_end].copy()

                detections = []

                # ----------------------------
                # TEMPLATE MATCHING (MULTI)
                # ----------------------------
                for name, template in templates:
                    th, tw = template.shape

                    if th > slice_img.shape[0] or tw > slice_img.shape[1]:
                        continue

                    res = cv2.matchTemplate(slice_img, template, METHOD)

                    # find ALL matches above threshold
                    ys, xs = np.where(res >= THRESHOLD)

                    for (x, y) in zip(xs, ys):
                        score = res[y, x]

                        detections.append({
                            "name": name,
                            "score": score,
                            "box": (x, y, tw, th)
                        })

                # ----------------------------
                # REMOVE DUPLICATES (NMS)
                # ----------------------------
                detections = non_max_suppression(detections)

                # ----------------------------
                # DRAW DETECTIONS
                # ----------------------------
                slice_copy = slice_color.copy()

                for det in detections:
                    x, y, tw, th = det["box"]

                    cv2.rectangle(
                        slice_copy,
                        (x, y),
                        (x + tw, y + th),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        slice_copy,
                        f"{det['name']} {det['score']:.2f}",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1
                    )

                # ----------------------------
                # SHOW SLICE
                # ----------------------------
                cv2.imshow("Slice", slice_copy)
                key = cv2.waitKey(0)

                # ----------------------------
                # SAVE MULTIPLE LABELS
                # ----------------------------
                if key == ord('y') and len(detections) > 0:

                    img_name = f"{save_index}.png"
                    label_name = f"{save_index}.txt"

                    img_path = os.path.join(OUTPUT_IMG_FOLDER, img_name)
                    label_path = os.path.join(OUTPUT_LABEL_FOLDER, label_name)

                    cv2.imwrite(img_path, slice_copy)

                    with open(label_path, "w") as f:
                        for det in detections:
                            x, y, tw, th = det["box"]

                            x_center = (x + tw / 2) / slice_copy.shape[1]
                            y_center = (y + th / 2) / slice_copy.shape[0]
                            w_norm = tw / slice_copy.shape[1]
                            h_norm = th / slice_copy.shape[0]

                            f.write(
                                f"{CLASS_ID} {x_center} {y_center} {w_norm} {h_norm}\n"
                            )

                    print(f"[SAVED] {img_name} ({len(detections)} objects)")
                    save_index += 1

                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

            cv2.destroyAllWindows()


if __name__ == "__main__":
    process()