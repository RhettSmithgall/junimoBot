import os
import time
import mss
import cv2
import numpy as np

# Create screenshots folder if it doesn't exist
SAVE_DIR = "screenshots"
os.makedirs(SAVE_DIR, exist_ok=True)

x=1

with mss.mss() as sct:
    # Capture full primary monitor
    monitor = {"top": 150, "left": 10, "width": 1500, "height": 825}
    while(True):
        screenshot = sct.grab(monitor)

        # Convert to numpy array (BGR for OpenCV)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Show the screenshot
        cv2.imshow("Slice", img)
        key = cv2.waitKey(0)

        if key == ord('y'):
            # Generate timestamp filename
            filename = f"{x}.png"
            filepath = os.path.join(SAVE_DIR, filename)
            x+=1

            cv2.imwrite(filepath, img)
            print(f"Saved to {filepath}")
        else:
            cv2.destroyAllWindows()

        cv2.destroyAllWindows()