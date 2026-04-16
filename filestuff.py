


def appendfile():
    file = open("negatives.txt")

    i = 1

    with open("negatives.txt", "a") as f:
        for x in range(0,500):
            line = f"negatives/{i}.jpg\n"
            f.write(line)
            i+=1

def fixfile():
    with open("positives.txt", "r") as f:
        lines = f.readlines()

    with open("positives_fixed.txt", "w") as f:
        for line in lines:
            f.write(line.replace("|", " "))

def yolo():
    import os
    import cv2

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)

    with open("positives.txt", "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(" ")

        img_path = parts[0]
        x = int(parts[2])
        y = int(parts[3])
        w = int(parts[4])
        h = int(parts[5])

        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Convert to YOLO format
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height

        # Save image
        filename = os.path.basename(img_path)
        new_img_path = f"dataset/images/train/{filename}"
        cv2.imwrite(new_img_path, img)

        # Save label
        label_path = f"dataset/labels/train/{filename.replace('.jpeg', '.txt')}"
        with open(label_path, "w") as f:
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}")
    return

yolo()