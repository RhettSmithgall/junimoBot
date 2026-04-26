import cv2
import mss
import numpy as np
import math
import os
import time
from ultralytics import YOLO


def main():

    with mss.mss() as sct:
        gameScreen = {
            "top": 240,
            "left": 100,
            "width": 1500,
            "height": 850
        }


        os.makedirs("positives", exist_ok=True)
        prev_time = 0
       

        # Load your specific custom model
        model = YOLO("runs/detect/train14/weights/best.pt")

        while True:

            screenshot = sct.grab(gameScreen)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
           

            # Prepare frame
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 1. Single-pass detection for player and coins
            player_pos, coin_positions = detect_game_objects(frame, model)

            # 2. Prioritize coins based on proximity to player
            sorted_coins = get_prioritized_coins(player_pos, coin_positions)

            # 3. Existing logic for tracks and lines
            tracks_pos = findlines(frame)

           

            # 4. Draw everything
            fullscreen = roboVision(player_pos, tracks_pos, sorted_coins, frame)

            cv2.putText(fullscreen, f"FPS: {int(fps)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Region Capture", fullscreen)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break


    cv2.destroyAllWindows()


def detect_game_objects(frame, model):

    """Detects minecart and coins in a single pass."""

    results = model(frame, verbose=False, stream=False)
    names = model.names
    #identify class ids based on models class names
    minecart_id = next((id for id, name in names.items() if name == "minecart"), None)
    coins_id = next((id for id, name in names.items() if name == "coins"), None)
    #tracks_id = next((id for id, name in names.items() if name == "tracks"), None)


    player_pos = (0, 0, 0, 0)
    coin_positions = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            #filter for minecart
            if cls_id == minecart_id:
                player_pos = tuple(map(int, box.xyxy[0]))
                #filter for coins
            elif cls_id == coins_id:
                coin_positions.append(tuple(map(int, box.xyxy[0])))

    return player_pos, coin_positions


def get_prioritized_coins(player_pos, coin_positions):
    """Sorts coins by distance from the player's center."""
    if player_pos == (0, 0, 0, 0) or not coin_positions:
        return coin_positions
    
    #extract player position
    px1, py1, px2, py2 = player_pos
    p_center = ((px1 + px2) / 2, (py1 + py2) / 2)

    def calculate_distance(coin):
        cx1, cy1, cx2, cy2 = coin
        #coin center cords
        c_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
        # Euclidean distance formula
        return math.sqrt((p_center[0] - c_center[0])**2 + (p_center[1] - c_center[1])**2)
    
    return sorted(coin_positions, key=calculate_distance)


def roboVision(player_pos, tracks_pos, sorted_coins, frame):
    # Create a copy of the frame for the transparency overlay
    overlay = frame.copy()
    # Draw track highlights on the OVERLAY
    for line in tracks_pos:
        lx1, ly1, lx2, ly2 = line
        # Draw a thick, neon-green line (BGR: 0, 255, 0)
        cv2.line(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 0), 6) # Increased thickness for "glow"
        
    # Blend the overlay with the original frame (0.4 is the transparency/alpha)
    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # extract player position
    px1, py1, px2, py2 = player_pos
    p_center = (int((px1 + px2) / 2), int((py1 + py2) / 2))

    if px1 != 0:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

    #draw colins and target line
    for i, (cx1, cy1, cx2, cy2) in enumerate(sorted_coins):
        c_center = (int((cx1 + cx2) / 2), int((cy1 + cy2) / 2))
        # draw closest coin
        if i == 0:
            color, label = (0, 0, 255), "TARGET"
            if px1 != 0:
                cv2.line(frame, p_center, c_center, (255, 255, 255), 2)
                cv2.putText(frame, label,(cx1,cy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            color, label = (0, 255, 255), "coin"
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 1)

    return frame



def get_angle(x1, y1, x2, y2):
    return abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))


def findlines(frame):
    img = frame
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
    return tracks


if __name__ == "__main__":
    main() 