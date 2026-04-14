import cv2
import mss
import numpy as np
import math

def main():
    with mss.mss() as sct:
        gameScreen = {
            "top": 140,
            "left": 10,
            "width": 1500,
            "height": 850
        }

        while True:
            #capture the game screen
            screenshot = sct.grab(gameScreen)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            playerShot = frame[10:850,10:650]

            output = frame.copy()

            fullscreen = findlines(frame,frame)

            fullscreen = findKart(playerShot,frame)

            scoreShot = frame[10:150,10:150]
            cv2.imshow("score", scoreShot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

            print(read_score(frame))

            cv2.imshow("Region Capture", fullscreen)

            if cv2.waitKey(1) & 0xFF == 27:  # press ESC to quit
                break

    cv2.destroyAllWindows()
    return

def read_score(frame):
    # Capture a specific area (left, top, right, bottom)
    reader = easyocr.Reader(['en'])

    # Extract only numbers from the image
    results = reader.readtext('screenshot.png', allowlist='0123456789')

    # Print detected numbers and their confidence scores
    for (bbox, text, prob) in results:
        print(f"Detected Number: {text} (Confidence: {prob:.4f})")

def findKart(screenshot,frame):
    method = cv2.TM_SQDIFF_NORMED

    # Read the images from the file
    small_image = cv2.imread('minecart.png')
    large_image = screenshot

    result = cv2.matchTemplate(large_image, small_image, method)

    # We want the minimum squared difference
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows,tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    #print(f"Detected position: x={MPx}, y={MPy}")
    cv2.putText(
    frame,
    "minecart",
    (MPx, MPy - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2)
    cv2.rectangle(frame, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

    # Display the original image with the rectangle around the match.
    #cv2.imshow('output',fullscreen)

    # The image is only displayed if we call this
    return frame

def get_angle(x1, y1, x2, y2):
    return abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

def showCartMask(img):
    cart_colors = [
        (0x1e, 0x30, 0x44),
        (0x17, 0x3d, 0x57),
        (0x65, 0x7c, 0x8e),
        (0x3f, 0x43, 0x56),
        (0x34, 0x28, 0x35),
        (0x32, 0x29, 0x36)
    ]

    tolerance = 25

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for (b, g, r) in cart_colors:
        lower = np.array([b - tolerance, g - tolerance, r - tolerance])
        upper = np.array([b + tolerance, g + tolerance, r + tolerance])

        color_mask = cv2.inRange(img, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)

    # apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # show debug windows
    cv2.imshow("mask", mask)
    cv2.imshow("filtered", result)

    return result

def findlines(screenshot,frame):
    img = screenshot
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

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = get_angle(x1, y1, x2, y2)
            length = np.hypot(x2 - x1, y2 - y1)
            
            if (length > 50) and (angle == 0 or angle == 45):
                #print(angle)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    return frame


main()