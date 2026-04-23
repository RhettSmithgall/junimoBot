import cv2
import numpy as np
import mss

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

# HSV sliders
cv2.createTrackbar("L - H", "Trackbars", 153, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 96, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 175, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 156, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Canny
cv2.createTrackbar("Thresh1", "Trackbars", 190, 1000, nothing)
cv2.createTrackbar("Thresh2", "Trackbars", 135, 1000, nothing)

# Hough
cv2.createTrackbar("hThresh", "Trackbars", 19, 400, nothing)
cv2.createTrackbar("hMinLine", "Trackbars", 20, 100, nothing)
cv2.createTrackbar("hMaxGap", "Trackbars", 1, 100, nothing)

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 400, 600)

kernel = np.ones((3,3), np.uint8)

with mss.mss() as sct:
    gameScreen = {"top": 150, "left": 10, "width": 1500, "height": 825}

    while True:
        # 🎛️ Read sliders
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        Thresh1 = cv2.getTrackbarPos("Thresh1", "Trackbars")
        Thresh2 = cv2.getTrackbarPos("Thresh2", "Trackbars")

        hThresh = cv2.getTrackbarPos("hThresh", "Trackbars")
        hMinLine = cv2.getTrackbarPos("hMinLine", "Trackbars")
        hMaxGap = cv2.getTrackbarPos("hMaxGap", "Trackbars")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])

        # 🎥 Capture frame
        screenshot = sct.grab(gameScreen)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 🎯 Process frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, kernel, iterations=1)

        edges = cv2.Canny(frame, Thresh1, Thresh2)
        edges[mask > 0] = 0  # remove masked areas

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=hThresh,
            minLineLength=hMinLine,
            maxLineGap=hMaxGap
        )

        # 🎨 Draw lines
        output = frame.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 🖥️ Show everything
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Edges", edges)
        cv2.imshow("Lines", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()