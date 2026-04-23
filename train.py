from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="trackTraining/dataset.yaml",
        epochs=100,
        imgsz=640,
        device=0
    )

if __name__ == "__main__":
    main()