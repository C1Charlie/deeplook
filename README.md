# Yolo 视觉识别项目
## Settings
    yolo settings runs_dir=path

## train
    yolo detect train data=./data_detect.yaml model=./runs/detect/train9/weights/best.pt epochs=60 imgsz=1920 device="mps"

## Detect

