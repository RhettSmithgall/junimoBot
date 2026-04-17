import mss
import numpy as np
import cv2

def find_ground_profile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape

    ground_y = np.full(w, -1, dtype=np.int32)

    step = 2  # skip pixels for speed

    for x in range(0, w, step):
        for y in range(h - 1, h // 3, -1):  # bottom → upward search
            if edges[y, x] > 0:
                ground_y[x] = y
                break

    return ground_y

with mss.mss() as sct:
    monitor = {"top": 150, "left": 10, "width": 1500, "height": 825}
    img = np.array(sct.grab(monitor))

    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    ground_y = find_ground_profile(frame)

    for x in range(len(ground_y)):
        y = ground_y[x]
        if y != -1:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow("ground", frame)
    cv2.waitKey(0)