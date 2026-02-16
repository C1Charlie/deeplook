import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==============================
# 配置
# ==============================

MODEL_PATH = "./runs/segment/train5/weights/best.pt"
CAM_INDEX = 0

CONF_TH = 0.25
N_FRAMES_CONFIRM = 3
AREA_RATIO_TH = 0.02   # 缺陷面积占比阈值（2%）

DEFECT_CLASSES = [1,2,3,4,5,6]  # 你的异常类别ID
NORMAL_CLASS = 0

# ==============================

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX)

history = deque(maxlen=N_FRAMES_CONFIRM)

print("🚀 Segment工业检测启动")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_area = h * w

    results = model.predict(frame, conf=CONF_TH, verbose=False)
    r = results[0]

    is_ng = False

    if r.boxes is not None and len(r.boxes) > 0:

        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ===== 使用 mask 计算面积 =====
            if r.masks is not None:
                mask = r.masks.data[i].cpu().numpy()
                mask_area = np.sum(mask > 0)
                area_ratio = mask_area / frame_area
            else:
                area_ratio = 0

            if cls in DEFECT_CLASSES:
                color = (0, 0, 255)
                label = f"DEFECT {conf:.2f}"
                is_ng = True
            else:
                color = (0, 255, 0)
                label = f"OK {conf:.2f}"

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 画 mask
            if r.masks is not None:
                mask_resized = cv2.resize(mask.astype(np.uint8)*255, (w, h))
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 2] = mask_resized  # 红色
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.3, 0)

            cv2.putText(frame,
                        f"{label} area={area_ratio:.3f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    # ===== 多帧确认 =====
    history.append(1 if is_ng else 0)
    confirmed_ng = (sum(history) == N_FRAMES_CONFIRM)

    # ===== 显示 OK/NG =====
    if confirmed_ng:
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        status = "NG"
        color = (0, 0, 255)
    else:
        status = "OK"
        color = (0, 255, 0)

    cv2.putText(frame,
                status,
                (int(w*0.4), int(h*0.2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                color,
                6)

    cv2.imshow("Segment Industrial Vision", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
