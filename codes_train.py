from ultralytics import YOLO
import torch


def main():
    # Проверка доступности GPU
    print(f"CUDA доступен: {torch.cuda.is_available()}")

    # Загрузка модели
    model = YOLO('yolov8s.pt')

    # Обучение
    results = model.train(
        data='training_data/codes_data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        patience=10,
        device=0 if torch.cuda.is_available() else 'cpu',
        name='test',
        plots=True,
        save=True,
        exist_ok=True,
        workers=0,
    )


if __name__ == '__main__':
    main()