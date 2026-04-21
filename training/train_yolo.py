import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a YOLOv8 detector.')
    parser.add_argument('--data', required=True, help='Path to dataset.yaml')
    parser.add_argument('--model', default='yolov8x.pt', help='YOLO model checkpoint')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size (CPU-friendly default)')
    parser.add_argument('--device', default='cpu', help='Training device: cpu or 0 for GPU')
    parser.add_argument('--workers', type=int, default=2, help='Dataloader workers')
    parser.add_argument('--project', default='training/runs', help='Output directory for runs')
    parser.add_argument('--name', default='yolov8x', help='Run name')
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
