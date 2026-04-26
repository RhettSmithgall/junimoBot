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
    ]

    for t in threads:
        t.daemon = True
        t.start()

    myprogress = progressTracker(0)

    model = DQN(input_size=5, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    #model = DQN(input_size=5, output_size=2)
    #model.load_state_dict(torch.load("dqn_model.pth"))
    #model.eval()

    epsilon = 1.0

    prev_progress = 0

    max_progress = 0
    max_level = 0

    step = 0

    while True:
        with lock:
            player = shared["player"]
            tracks = shared["tracks"]
            progress = shared["progress"]
            attempt = shared["attempt"]

        myprogress.updateProgress(progress)

        state = get_state(player, tracks, progress)

        if state is None:
            continue

        state_t = torch.tensor(state)

        # ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)
        else:                        
            with torch.no_grad():
                q_vals = model(state_t)
                action = torch.argmax(q_vals).item()

        # do action
        do_action(action)

        # get next state
        with lock:
            new_progress = shared["progress"]
            new_attempt = shared["attempt"]
            player2 = shared["player"]
            tracks2 = shared["tracks"]

        myprogress.updateProgress(progress)

        died = new_attempt != attempt

        next_state = get_state(player2, tracks2, new_progress)
        if next_state is None:
            continue

        reward = compute_reward(prev_progress, new_progress, died)

        prev_progress = new_progress

        # --- TRAIN ---
        next_state_t = torch.tensor(next_state)

        q_vals = model(state_t)
        q_val = q_vals[action]

        with torch.no_grad():
            next_q_vals = model(next_state_t)
            max_next_q = torch.max(next_q_vals)

        target = reward + (0.99 * max_next_q * (0 if died else 1))

        loss = loss_fn(q_val, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if died:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            prev_progress = 0   
            print(f"Run Number: {attempt} Progress achieved: {progress}")                      

        # decay exploration
        epsilon = max(0.1, epsilon * 0.995)  

        with lock:
            if(progress > max_progress):
                max_progress = progress 
                print(f"Furthest distance: {max_progress}") 

        if step == 1000:
            torch.save(model.state_dict(), "dqn_model.pth")  
            step = 0

        step+=1

def compute_reward(prev_progress, current_progress, died):
    reward = (current_progress - prev_progress) * 0.1

    if died:
        reward -= 10  # big punishment for failing

    return reward

def do_action(action):
    if action == 1:
        pyautogui.keyDown("space")
    else:
        pyautogui.keyUp("space")

def get_state(player, tracks, progress):
    if player is None or tracks is None:
        return None

    px, py = player

    nearest_y = py
    slope = 0

    min_dist = 99999

    for box in tracks:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cy = (y1 + y2) // 2
        dist = abs(cy - py)

        if dist < min_dist:
            min_dist = dist
            nearest_y = cy
            slope = int(box.cls[0])  # 0 flat, 1 down, 2 up

    return np.array([
        px / 1500,
        py / 825,
        nearest_y / 825,
        slope / 2,
        progress / 800
    ], dtype=np.float32)

#what run number are we on? What level?
class progressTracker:
    prev = 0

    def __init__(self, progress):
        self.progress = progress

    def updateProgress(self, progress):
        self.progress = progress

        if(progress < 40):
            return
        
        if(progress+10 < self.prev and self.prev < 790):
            #reset
            self.prev = 0
            with lock:
                shared["attempt"] += 1
                shared["level"] = 0
            return True
        
        if(progress < self.prev) and self.prev > 790:
            #end of a level!
            self.prev = 0
            with lock:
                shared["level"] += 1
            return False

        if(progress >= self.prev):
            #normal progression
            self.prev = progress

        return False

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
