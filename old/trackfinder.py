import cv2
import numpy as np
import os
import mss

# SETTINGS
TEMPLATE_FOLDER = "tracks"
OUTPUT_IMG_FOLDER = "trackTraining\\train"
OUTPUT_LABEL_FOLDER = "trackTraining\\val"
NUM_SLICES = 10
METHOD = cv2.TM_CCOEFF_NORMED
THRESHOLD = 0.6  # minimum confidence

CLASS_ID = 2  # change if you have multiple classes


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


def save_yolo_label(path, class_id, x, y, w, h, img_w, img_h):
    # convert to YOLO format (normalized center coords)
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    with open(path, "w") as f:
        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")


def process():

    save_index = 323

    while(True):
        with mss.mss() as sct:
            monitor = {"top": 150, "left": 10, "width": 1500, "height": 825}

            sct_img = sct.grab(monitor)
            ensure_dirs()

            frame = np.array(sct_img)[:, :, :3]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            slice_width = w // NUM_SLICES

            templates = load_templates(TEMPLATE_FOLDER)

            for i in range(NUM_SLICES):
                x_start = i * slice_width
                x_end = x_start + slice_width

                slice_img = gray[:, x_start:x_end]
                slice_color = frame[:, x_start:x_end].copy()

                best_score = -1

                for name, template in templates:
                    th, tw = template.shape

                    if th > slice_img.shape[0] or tw > slice_img.shape[1]:
                        continue

                    res = cv2.matchTemplate(slice_img, template, METHOD)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    if max_val > best_score:
                        best_score = max_val
                        best_name = name
                        best_box = (max_loc[0], max_loc[1], tw, th)

                if best_score < THRESHOLD:
                    continue


                # draw detection if confident
                if best_box is not None and best_score > THRESHOLD:
                    slice_copy = slice_color.copy()
                    x, y, tw, th = best_box

                    cv2.rectangle(slice_color, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                    cv2.putText(
                        slice_color,
                        f"{best_name} {best_score:.2f}",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1
                    )

                # show slice
                cv2.imshow("Slice", slice_color)
                key = cv2.waitKey(0)

                if key == ord('y') and best_box is not None and best_score > THRESHOLD:
                    x, y, tw, th = best_box

                    img_name = f"{save_index}.png"
                    label_name = f"{save_index}.txt"

                    img_path = os.path.join(OUTPUT_IMG_FOLDER, img_name)
                    label_path = os.path.join(OUTPUT_LABEL_FOLDER, label_name)

                    # save image
                    cv2.imwrite(img_path, slice_copy)

                    # save label
                    save_yolo_label(label_path, CLASS_ID, x, y, tw, th,
                                    slice_color.shape[1], slice_color.shape[0])

                    print(f"[SAVED] {img_name}")

                    save_index += 1

                elif key == ord('q'):
                    break

            cv2.destroyAllWindows()


if __name__ == "__main__":
    process()