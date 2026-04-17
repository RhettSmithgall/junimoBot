import time
import mss
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import math

track_templates = [
    cv2.imread("tracks/track1.png", 0),
    cv2.imread("tracks/track2.png", 0),
    cv2.imread("tracks/track3.png", 0),
    cv2.imread("tracks/track4.png", 0)
]

shared = {
    "frame": None,
    "player": None,
    "tracks": None,
    "barricades": None
    }

lock = threading.Lock()
running = True

model = YOLO("runs/detect/train2/weights/best.pt")

def main():
    threads = [
        threading.Thread(target=capture_loop),
        threading.Thread(target=find_kart),
        threading.Thread(target=find_tracks)#,
        #threading.Thread(target=find_barricades),
    ]
    
    for t in threads:
        t.daemon = True
        t.start()

    last_time = time.time()
    frames = 0

    while True:
        with lock:
            player = shared["player"]
            tracks = shared["tracks"]
            frame = shared["frame"]

            if frame is None:
                continue

            frame = frame.copy()

        if(roboVision(frame, player, tracks)):
            break

        frames += 1
        if time.time() - last_time >= 1:
            print("Loop FPS:", frames)
            frames = 0
            last_time = time.time()

#see what the robot sees
def roboVision(frame, player, tracks):
    #draw minecart
    if player is not None:
        cx, cy = player

        box_w = 50
        box_h = 50

        x1 = cx - box_w // 2
        y1 = cy - box_h // 2
        x2 = cx + box_w // 2
        y2 = cy + box_h // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Minecart",
            (x1, y1 - 8),  # slightly above box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    #draw tracks
    if tracks is not None:
        for line in tracks:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Game", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
        return True
    
    return False

def capture_loop():
    with mss.mss() as sct:
        monitor = {"top": 150, "left": 10, "width": 1500, "height": 825}

        while running:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)[:, :, :3]

            with lock:
                shared["frame"] = frame

def find_tracks():
    while running:
        with lock:
            if shared["frame"] is None:
                continue
            
            thisFrame = shared["frame"].copy()

        thisFrame = cv2.cvtColor(thisFrame, cv2.COLOR_BGRA2BGR)

        img = thisFrame
        # --- 3 track colors (hex -> BGR) ---
        colors = [
            (0xA7, 0xD3, 0xE7),  # e7d3a7
            (0x80, 0x94, 0xAF),  # af9480
            (0xAB, 0xBE, 0xCF)   # cfbeab
        ]

        tolerance = 20

        # --- build mask ---
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for (b, g, r) in colors:
            lower = np.array([b - tolerance, g - tolerance, r - tolerance])
            upper = np.array([b + tolerance, g + tolerance, r + tolerance])

            color_mask = cv2.inRange(img, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)

        # --- clean noise ---
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- edges ---
        edges = cv2.Canny(mask, 50, 150)

        # --- hough lines ---
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=40,
            maxLineGap=10
        )

        tracks = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = get_angle(x1, y1, x2, y2)
                length = np.hypot(x2 - x1, y2 - y1)
                
                if (length > 50) and (angle == 0 or angle == 45):
                    #print(angle)
                    tracks.append((x1, y1, x2, y2))

        with lock:
            shared["tracks"] = tracks
    
def get_angle(x1, y1, x2, y2):
    return abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

def find_kart():
    while running:
        with lock:
            frame = shared["frame"]

        if frame is None:
            continue

        thisFrame = frame.copy()

        #only look at a small slice of the screen, for speed
        crop = thisFrame[:, 270:430]

        #call upon the power of yolo
        results = model.predict(crop, conf=0.5, verbose=False)

        pos = None
        for box in results[0].boxes:
            #get the coordinates of the player
            cords = box.xyxy[0].tolist()
            #simplify these to a single point
            x1, y1, x2, y2 = cords

            cx = int(((x1+270) + (x2+270)) / 2)
            cy = int((y1 + y2) / 2)

            pos = (cx, cy)
            break

        with lock:
            shared["player"] = pos

main()
