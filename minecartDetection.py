import cv2
import mss
import numpy as np
import math
import os
from ultralytics import YOLO

def main():
    with mss.mss() as sct:
        gameScreen = {
            "top": 140,
            "left": 10,
            "width": 1500,
            "height": 850
        }

        # make sure positives folder exists
        os.makedirs("positives", exist_ok=True)

        img_counter = 0

        model = YOLO("runs/detect/train2/weights/best.pt")

        while True:
            #capture the game screen
            screenshot = sct.grab(gameScreen)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            playerShot = frame[10:850,200:500]

            trackshot = frame[100:900,400:1200]

            player_pos = detect_minecart(playerShot,model)

            tracks_pos = findlines(trackshot)

            barricade_pos = findBarricades(trackshot)

            print(f"player: {player_pos} tracks: {len(tracks_pos)} barricade?: {barricade_pos}")

            #fullscreen = roboVision(player_pos,tracks_pos,barricade_pos, frame)

            #cv2.imshow("Region Capture", fullscreen)

            if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
                break

    cv2.destroyAllWindows()
    return

#see what the robot sees
def roboVision(player_pos, tracks_pos, barriacade_pos, frame):
    
    x1, y1, x2, y2 = player_pos
    x1 +=200
    x2 +=200
    # draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

    # label
    cv2.putText(frame, "minecart", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    for line in tracks_pos:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    MPx, MPy = barriacade_pos
    if(MPx != 0 and MPy != 0):
        cv2.putText(frame,"Barrricade",(MPx, MPy - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),2)
        cv2.rectangle(frame, (MPx,MPy),(MPx+40,MPy+60),(255,0,0),2)

    return frame

def detect_minecart(frame,model):
    results = model(frame, max_det=1, verbose=False, stream=False)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            return x1, y1, x2, y2
    return 0,0,0,0

def findBarricades(frame):
    method = cv2.TM_SQDIFF_NORMED

    # Read the images from the file
    small_image = cv2.imread('barricade.png')
    large_image = frame

    result = cv2.matchTemplate(large_image, small_image, method)

    # We want the minimum squared difference
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)

    confidence = 1 - mn  # convert to "higher is better"

    if confidence < 0.7:
        return 0,0

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = mnLoc

    # The image is only displayed if we call this
    return MPx,MPy

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

main()