import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==============================
# 配置区
# ==============================

MODEL_PATH = "./runs/detect/train7/weights/best.pt"  # 改成你的模型路径
CAM_INDEX = 0

CONF_TH = 0.5              # 置信度阈值
N_FRAMES_CONFIRM = 1     # 连续几帧判定才算 NG
DISPLAY_SCALE = 1.0        # 画面缩放

# 如果你的类别如下：
# 0 = 正常
# 其他 = 异常
NORMAL_CLASS_ID = 0

# ==============================
# 初始化
# ==============================

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

history = deque(maxlen=N_FRAMES_CONFIRM)

print("🚀 工业检测系统启动...")
print("按 ESC 退出")

# ==============================
# 主循环
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取摄像头失败")
        break

    if DISPLAY_SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    h, w = frame.shape[:2]

    # ==============================
    # YOLO 推理
    # ==============================

    results = model.predict(frame, conf=CONF_TH, verbose=False)
    r = results[0]

    print("results:",r.boxes)

    is_ng = False

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # if cls == NORMAL_CLASS_ID:
            #     color = (0, 255, 0)
            #     label = f"OK {conf:.2f}"
            # else:
            color = (0, 0, 255)
            label = f"DEFECT {conf:.2f}"
            is_ng = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    # ==============================
    # 多帧确认
    # ==============================

    history.append(1 if is_ng else 0)
    confirmed_ng = (sum(history) == N_FRAMES_CONFIRM)

    # ==============================
    # 画面状态显示
    # ==============================

    if confirmed_ng:
        status_text = "NG"
        color = (0, 0, 255)

        # 整屏淡红
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    else:
        status_text = "OK"
        color = (0, 255, 0)

    # 中央大字
    cv2.putText(frame,
                status_text,
                (int(w * 0.4), int(h * 0.2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                color,
                6)

    # 左上角小信息
    cv2.putText(frame,
                f"CONF_TH={CONF_TH}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.imshow("Industrial Vision System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# ==============================
# 释放资源
# ==============================

cap.release()
cv2.destroyAllWindows()
print("系统关闭")
