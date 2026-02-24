import cv2
import numpy as np
import time
from ultralytics import YOLO

# =========================
# 🔧 可调参数区域
# =========================
MODEL_PATH = "./runs/detect/train8/weights/best.pt"

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
FRAME_CONFIRM = 3      # 连续多少帧才判定 NG
SHOW_CONF = True

WIDTH = 1280
HEIGHT = 720

# =========================
# 加载模型
# =========================
model = YOLO(MODEL_PATH)

# =========================
# 打开摄像头
# =========================
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

for cam in [cam0, cam1]:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# 连续帧计数
ng_counter_0 = 0
ng_counter_1 = 0

# FPS 计算
prev_time = time.time()

# =========================
# 主循环
# =========================
while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0 or not ret1:
        break

    # YOLO 推理
    results0 = model(frame0, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    results1 = model(frame1, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

    def process(frame, results, ng_counter):
        anomaly_count = 0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                anomaly_count += 1

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                text = label
                if SHOW_CONF:
                    text += f" {conf:.2f}"

                cv2.putText(frame, text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255), 2)

        # 延迟确认机制
        if anomaly_count > 0:
            ng_counter += 1
        else:
            ng_counter = 0

        confirmed_ng = ng_counter >= FRAME_CONFIRM

        # 状态显示
        status = "NG" if confirmed_ng else "OK"
        color = (0, 0, 255) if confirmed_ng else (0, 255, 0)

        cv2.putText(frame,
                    f"Anomaly: {anomaly_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        cv2.putText(frame,
                    status,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    4)

        return frame, ng_counter

    frame0, ng_counter_0 = process(frame0, results0, ng_counter_0)
    frame1, ng_counter_1 = process(frame1, results1, ng_counter_1)

    # 计算 FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    combined = np.hstack((frame0, frame1))

    cv2.putText(combined,
                f"FPS: {fps:.2f}",
                (10, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.imshow("Dual Camera Industrial Inspection", combined)

    # =========================
    # 🎛️ 快捷调试按键