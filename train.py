from ultralytics import YOLO

# 1. Load a pretrained model (e.g., YOLO11 nano is fast and lightweight)
model = YOLO("yolo11n.pt") 

# 2. Train the model
results = model.train(
    data="path/to/data.yaml", 
    epochs=100, 
    imgsz=640, 
    device=0  # Use 0 for GPU, 'cpu' for CPU
)
