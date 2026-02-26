# Yolo 视觉识别项目
## Settings
    yolo settings runs_dir=path

## train
    yolo detect train data=./data_detect.yaml model=./runs/detect/train9/weights/best.pt epochs=60 imgsz=1920 device="mps"
### 在训练时出现了内存限制我们可以使用一下方法避免
    yolo detect train \
    data=./data_detect.yaml \
    model=./yolov8n.pt \
    epochs=60 \
    imgsz=1280 \
    batch=4 \
    device=mps
batch=4
控制显存

    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0


### ① train/box_loss
👉 边框回归损失
含义：
模型预测的框和真实框之间的位置误差。
越低越好。
你这张图的情况：
    从 ~0.85 下降到 ~0.52
持续下降，没有反弹
✅ 说明模型定位能力在稳定提升
✅ 没有明显过拟合迹象


### train/cls_loss

👉 分类损失

表示模型在判断类别（OK / NG / 缺陷类型）上的错误程度。
越低越好。

你这张图：

从 3.2 左右下降到 1.7 左右

有波动但整体下降

✅ 分类能力明显提升
⚠️ 中间有一点波动（可能样本少）

### train/dfl_loss

👉 Distribution Focal Loss（边框精细定位）

YOLOv8 用来做更精确边界框拟合的损失。

越低越好。

你图：

从 1.05 → 0.89

稳定下降

✅ 定位精度在提升
工业缺陷检测中这个很重要


## Detect


