# Offline Training (COCO JSON)

This folder trains a high-accuracy YOLOv8 detector from COCO-style JSON.

## 1) Install Python deps
```
pip install -r training/requirements.txt
```

## 2) Convert COCO JSON to YOLO labels
```
python training/prepare_coco_yolo.py \
  --train-json "C:\\path\\to\\annotations\\instances_train.json" \
  --val-json "C:\\path\\to\\annotations\\instances_val.json" \
  --train-images "C:\\path\\to\\images\\train" \
  --val-images "C:\\path\\to\\images\\val" \
  --output-dir "C:\\path\\to\\output\\navassist-yolo"
```

This will copy images into the output folder and write YOLO labels + dataset.yaml.

## 3) Train the model (accuracy-first)
```
python training/train_yolo.py \
  --data "C:\\path\\to\\output\\navassist-yolo\\dataset.yaml" \
  --model yolov8x.pt \
  --device cpu
```

Notes:
- On an Intel i5 CPU, YOLOv8x can take days to train. This is expected.
- If you have a compatible GPU, set `--device 0` to use it.
