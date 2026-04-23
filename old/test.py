from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-4/weights/best.pt")

results = model("screenshot3.png", conf=0.25)

annotated = results[0].plot()

cv2.imshow("YOLO Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()