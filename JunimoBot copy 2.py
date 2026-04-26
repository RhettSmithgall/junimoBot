import time
import mss
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import math
import pyautogui
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

shared = {
    "frame": None,
    "player": None,
    "tracks": None,
    "gaps": None,
    "barricades": None,
    "progress": 0,
    "level": 0,
    "attempt": 0
    }

lock = threading.Lock()
running = True

Kart_model = YOLO("runs/detect/train2/weights/best.pt")

track_model = YOLO("runs/detect/train-4/weights/best.pt")

def main():
    threads = [
        threading.Thread(target=capture_loop),
        threading.Thread(target=find_kart),
        threading.Thread(target=find_tracks),
        threading.Thread(target=find_progress)
        #threading.Thread(target=find_barricades),
    ]
    
    for t in threads:
        t.daemon = True
        t.start()

    last_time = time.time()
    frames = 0

    with lock:
        myProgress = progressTracker(shared["progress"])

    while True:
        with lock:
            player = shared["player"]
            tracks = shared["tracks"]
            frame = shared["frame"]
            progress = shared["progress"]

            if frame is None:
                continue

            frame = frame.copy()

        myProgress.updateProgress(progress)

        action(player,tracks)

        #if(roboVision(frame, player, tracks, progress)):
           #break

        frames += 1
        if time.time() - last_time >= 1:
            #print("Loop FPS:", frames)
            frames = 0
            last_time = time.time()

def action(player, tracks):

    return

def get_current_track(player, tracks):
    player_x,player_y = player

    for t in tracks:
        x, y, w, h = t
        
        if x <= player_x <= x + w:
            if abs(player_y - y) < 50:  # tolerance
                return t
    return None

#what run number are we on? What level?
class progressTracker:
    prev = 0

    def __init__(self, progress):
        self.progress = progress

    def updateProgress(self, progress):
        self.progress = progress

        if(progress < 40):
            return

        print(f"OUT: {progress} {self.prev}")
        if(progress+10 < self.prev and self.prev < 790):
            #reset
            self.prev = 0
            with lock:
                shared["attempt"] += 1
                shared["level"] = 0
            return
        
        if(progress < self.prev) and self.prev > 790:
            #end of a level!
            self.prev = 0
            with lock:
                shared["level"] += 1
            return

        if(progress >= self.prev):
            #normal progression
            self.prev = progress

#see how far in a run the kart is
def find_progress():
    while running:
        with lock:
            if shared["frame"] is None:
                continue

            thisFrame = shared["frame"].copy()

        #only look at a small slice of the screen, for speed
        crop = thisFrame[30:90, 400:1300]

        # Convert it to grayscale
        img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Read the template
        template = cv2.imread("progress_kart_template.png", 0)

        # Store width and height of template in w and h
        w = 40
        h = 45

        # Perform match operations.
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Specify a threshold
        threshold = 0.7

        loc = np.where(res >= threshold)

        x = None

        for pt in zip(*loc[::-1]):
            x = pt[0]

        if x is None:
            x = 0

        with lock:
            shared["progress"] = x
        
#see what the robot sees
def roboVision(frame, player, tracks, progress):
    #frame = np.zeros((825, 1500, 3), dtype="uint8")

    #draw minecart
    if player is not None:
        cx, cy = player

        box_w = 50
        box_h = 50

        x1 = cx - box_w // 2
        y1 = cy - box_h // 2
        x2 = cx + box_w // 2
        y2 = cy + box_h // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 0,255), 2)
        cv2.putText(frame,"Minecart",(x1, y1 - 8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)

    with lock:
        level = shared["level"]
        attempt = shared["attempt"]

    text = f"LEVEL: {level} PROGRESS: {progress}  RUN #: {attempt}"
    cv2.putText(frame, text, (750 - len(text)*10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)

    # --- draw platforms (GREEN) ---
    if tracks is not None:
        for box in tracks:
            # xyxy = (x1, y1, x2, y2) in PIXELS already
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            label = f"{cls} {conf:.2f}"

            y1+=20

            if cls == 0:
                # flat
                cv2.line(frame, (x1, y1), (x2, y1), (255, 0, 0), 3)

            elif cls == 1:
                # slope down
                cv2.line(frame, (x1, y1), (x2, y1+60), (255, 0, 0), 2)

            elif cls == 2:
                # slope up
                cv2.line(frame, (x1, y1+60), (x2, y1), (255, 0, 0), 2)


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
            frame = shared["frame"]

        if frame is None:
            continue

        thisFrame = frame.copy()

        results = track_model(thisFrame, conf=0.25, verbose=False)[0]

        boxes = results.boxes

        if boxes is None:
            continue

        with lock:
            shared["tracks"] = boxes

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
        results = Kart_model.predict(crop, conf=0.5, verbose=False)

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
