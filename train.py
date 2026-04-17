from ultralytics import YOLO

model = YOLO("yolo11n.pt") 

results = model.train(
    data="path/to/data.yaml", 
    epochs=100, 
    imgsz=640, 
    device=0  
)
