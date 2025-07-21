from ultralytics import YOLO
import torch


def main():
    print(f"CUDA доступен: {torch.cuda.is_available()}")

    model = YOLO('yolov8s.pt')

    results = model.train(
        data='training_data/frame_data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        patience=10,
        device=0 if torch.cuda.is_available() else 'cpu',
        name='frame_detection',
        plots=True,
        save=True,
        exist_ok=True,
        workers=0
    )


if __name__ == '__main__':
    main()