import cv2
import mss
import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from ultralytics import YOLO
import pydirectinput
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pytesseract

# --- Configuration ---
# 1. OCR Setup
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DEBUG = False  # Toggle this to True if you need to see the visual windows again

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

# Game Region Setup
SCREEN_REGION = {"top": 140, "left": 100, "width": 1500, "height": 850} 
# score region setup
SCORE_REGION = (250, 10, 120, 35) # (x, y, w, h)

# class ids from data.yaml
ID_BARRICADE = 0
ID_BOULDER = 1
ID_CHERRY = 3
ID_COINS = 4
ID_MINECART = 5
ID_TRACKS = 7

class JunimoKartEnv(gym.Env):
    def __init__(self):
        '''
        Gym Environment for Jumino Kart
        launches browser instance and loads yolo model
        '''
        super(JunimoKartEnv, self).__init__()
        # Actions 0: no operation, 1: light press 2: long press
        self.action_space = spaces.Discrete(3)
        # Observation space for 6 class ids witgh x,y cords
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        # Selenium browser setup
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1600,1000")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.get("https://thomaswp.github.io/JunimoKartWeb/Source/index.html?mode=2&level=0")
        # Loading yolo model, currently on run 14
        self.model = YOLO("runs/detect/train14/weights/best.pt")
        self.prev_dist = float('inf')
        self.last_score = 0
        
        time.sleep(5) 

    def get_score(self, frame):
        '''Extracts game score from frame'''
        if not OCR_AVAILABLE: return 0
        try:
            x, y, w, h = SCORE_REGION
            score_roi = frame[y:y+h, x:x+w]
            
            # Pre-processing, upscale for clarity and threshold for contrast
            upscaled = cv2.resize(score_roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(upscaled, 150, 255, cv2.THRESH_BINARY_INV)
            
            # --- LIVE REFRESH DEBUG ---
            if DEBUG:
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Main Debug View", debug_frame)
                cv2.imshow("Tesseract sees this", thresh)
                cv2.waitKey(10)
            
            # config to only look for digits
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            # Print to console only if we detect a number
            if text:
                print(f"OCR Detected Score: {text}")
                
            return int(text) if text else self.last_score
        except:
            return self.last_score

    def _get_obs(self):
        '''Captures the game screen and uses Yolo to define the environemnt state'''
        with mss.mss() as sct:
            screenshot = sct.grab(SCREEN_REGION)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            results = self.model(frame, verbose=False)
            
            
            # Initialize with default values (0, 0)
            player_pos = [0, 0]
            nearest_objs = {ID_BARRICADE: [0, 0], ID_BOULDER: [0, 0], 
                        ID_CHERRY: [0, 0], ID_COINS: [0, 0], ID_TRACKS: [0, 0]}
            # Temporary storage for calculating nearest
            detected_positions = {ID_BARRICADE: [], ID_BOULDER: [], 
                              ID_CHERRY: [], ID_COINS: [], ID_TRACKS: []}
            
            debug_frame = frame.copy() if DEBUG else None
            # iterate throguh all detected boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls)   
                    conf = float(box.conf)
                    
                    #filter out from maping objects in the ui (specifically tracks being hallucinated as part of the background)
                    if y1 < 100:
                        continue
                    #center coordinates
                    center = [((x1+x2)/2)/1500, ((y1+y2)/2)/850]
                    # identify player or store locaion of object
                    if cls_id == ID_MINECART:
                        player_pos = center
                    elif cls_id in detected_positions:
                        detected_positions[cls_id].append(center)
                        
                        
                     # debug to draw bounding boxes over objects detected   
                    if DEBUG:
                        class_names = ['barricade', 'boulder', 'bubble', 'cherry', 'coins', 'minecart', 'score', 'tracks']
                        conf = float(box.conf)
                        label = f"{class_names[cls_id]} {conf:.2f} {y1}"
                        # Draw boxes for items that passed the filter
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if DEBUG:
                cv2.imshow("AI Vision Debug (Filtered)", debug_frame)
                cv2.waitKey(1)

            # Find nearest object for each category
            for cls_id, positions in detected_positions.items():
                if positions:
                    # Calculate Euclidean distance to player
                    sorted_pos = sorted(positions, key=lambda c: math.sqrt((player_pos[0]-c[0])**2 + (player_pos[1]-c[1])**2))
                    nearest_objs[cls_id] = sorted_pos[0]

            # Flatten into a single observation array
            obs = np.array(player_pos + 
                       nearest_objs[ID_BARRICADE] + 
                       nearest_objs[ID_BOULDER] + 
                       nearest_objs[ID_CHERRY] +  
                       nearest_objs[ID_COINS] + 
                       nearest_objs[ID_TRACKS], dtype=np.float32)
            
            return np.nan_to_num(obs, nan=0.0), frame

    def step(self, action):
        # 0: Do nothing (No-Op)
        # 1: Quick Press
        # 2: Long Press
        
        if action == 1: 
            pydirectinput.press('space')
        elif action == 2:
            pydirectinput.keyDown('space')
            time.sleep(0.2)
            pydirectinput.keyUp('space')
        # Else (action == 0): do nothing, implicitly handled
            
        obs, frame = self._get_obs()
        
        # Termination check (Score-based)
        current_score = self.get_score(frame)
        terminated = False
        if self.last_score > 0 and current_score < self.last_score:
            terminated = True
        
        if terminated:
            reward = -1.0 
        else:
            # Reward logic, positive reward fo moving towards objects (coins) and increasing score
            current_dist = np.linalg.norm(obs[0:2] - obs[2:4])
            dist_reward = 0 if self.prev_dist == float('inf') else np.clip((self.prev_dist - current_dist) * 10, -1.0, 1.0)
            score_reward = max(0, (current_score - self.last_score) * 0.5)
            
            reward = dist_reward + score_reward
            self.prev_dist = current_dist
            self.last_score = current_score
        
        return obs, reward, terminated, False, {}
        

    def reset(self, seed=None, options=None):
        '''Resets the environment variables'''
        self.prev_dist = float('inf')
        self.last_score = 0
        obs, _ = self._get_obs()
        return obs, {}

    def close(self):
        '''closes resources on shutdown'''
        if DEBUG: cv2.destroyAllWindows()
        self.driver.quit()

if __name__ == "__main__":
    #initialize training
    env = JunimoKartEnv()
    #save a checkpoint ever 10k steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="junimo_kart")
    # PPO algoritm with multilayer perception policy
    model = PPO("MlpPolicy", env, verbose=1)
    #start training
    model.learn(total_timesteps=500000, callback=checkpoint_callback)
    env.close()