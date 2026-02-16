import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

MODEL_PATH = "./runs/segment/train6/weights/best.pt"  # 改成你的 best.pt
CAM_INDEX = 0

# YOLO判定（也做成滑条了）
DEFAULT_CONF_TH = 50   # 0~100 -> 0.50
DEFAULT_COVER_TH = 40  # 0~100 -> 0.40

N_FRAMES_CONFIRM = 3
PADDING = 0.08

model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(0)

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break
#     cv2.imshow("iphone", frame)
#     if cv2.waitKey(1) == 27:
#         break

foreign_history = deque(maxlen=N_FRAMES_CONFIRM)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def nothing(x):
    pass

# ---------- UI: Trackbars ----------
cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("controls", 520, 260)

# 白色判定阈值（你要调的核心）
cv2.createTrackbar("WHITE_V_MIN", "controls", 200, 255, nothing)   # 180~230常用
cv2.createTrackbar("WHITE_S_MAX", "controls", 60, 255, nothing)    # 40~120常用

# 去噪与过滤
cv2.createTrackbar("MIN_AREA", "controls", 2500, 50000, nothing)   # 按分辨率调
cv2.createTrackbar("KERNEL", "controls", 7, 25, nothing)           # 5/7/9/11...奇数更好
cv2.createTrackbar("OPEN_IT", "controls", 1, 5, nothing)
cv2.createTrackbar("CLOSE_IT", "controls", 2, 8, nothing)

# YOLO判定阈值
cv2.createTrackbar("CONF_TH(%)", "controls", DEFAULT_CONF_TH, 100, nothing)
cv2.createTrackbar("COVER_TH(%)", "controls", DEFAULT_COVER_TH, 100, nothing)

# 可选：显示模式（0:frame, 1:mask, 2:mask_on_frame）
cv2.createTrackbar("VIEW(0/1/2)", "controls", 0, 2, nothing)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read camera frame.")
        break

    h, w = frame.shape[:2]

    # ----- 读滑条参数 -----
    WHITE_V_MIN = cv2.getTrackbarPos("WHITE_V_MIN", "controls")
    WHITE_S_MAX = cv2.getTrackbarPos("WHITE_S_MAX", "controls")

    MIN_AREA = cv2.getTrackbarPos("MIN_AREA", "controls")
    K = cv2.getTrackbarPos("KERNEL", "controls")
    OPEN_IT = cv2.getTrackbarPos("OPEN_IT", "controls")
    CLOSE_IT = cv2.getTrackbarPos("CLOSE_IT", "controls")

    CONF_TH = cv2.getTrackbarPos("CONF_TH(%)", "controls") / 100.0
    COVER_TH = cv2.getTrackbarPos("COVER_TH(%)", "controls") / 100.0

    view_mode = cv2.getTrackbarPos("VIEW(0/1/2)", "controls")

    if K < 1:
        K = 1
    if K % 2 == 0:
        K += 1

    # ----- 白纸背景：找“非白色区域” -----
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_mask = ((V >= WHITE_V_MIN) & (S <= WHITE_S_MAX)).astype(np.uint8) * 255
    obj_mask = cv2.bitwise_not(white_mask)

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    if OPEN_IT > 0:
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_IT)
    if CLOSE_IT > 0:
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_IT)

    # 找轮廓 ROI
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(1, MIN_AREA):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        pad_x = int(bw * PADDING)
        pad_y = int(bh * PADDING)
        x1 = clamp(x - pad_x, 0, w - 1)
        y1 = clamp(y - pad_y, 0, h - 1)
        x2 = clamp(x + bw + pad_x, 0, w - 1)
        y2 = clamp(y + bh + pad_y, 0, h - 1)
        rois.append((x1, y1, x2, y2))

    foreign_count = 0

    # ROI -> YOLO 判断 A01（你只有一个类，所以 cls==0）
    for (x1, y1, x2, y2) in rois:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        results = model.predict(roi, conf=CONF_TH, verbose=False)
        r = results[0]

        is_main = False
        best_cover = 0.0
        best_conf = 0.0

        if r.boxes is not None and len(r.boxes) > 0:
            roi_area = max(1, (x2 - x1) * (y2 - y1))
            for b in r.boxes:
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                if cls != 0:
                    continue
                xA, yA, xB, yB = b.xyxy[0].cpu().numpy()
                box_area = max(1, (xB - xA) * (yB - yA))
                cover = box_area / roi_area
                if cover > best_cover:
                    best_cover = cover
                    best_conf = conf

            if best_cover >= COVER_TH:
                is_main = True

        if is_main:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"MAIN A01 conf={best_conf:.2f} cov={best_cover:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        else:
            foreign_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "FOREIGN",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

    # 多帧确认
    foreign_history.append(1 if foreign_count > 0 else 0)
    alarm = (sum(foreign_history) == N_FRAMES_CONFIRM)

    status = "ALARM: FOREIGN" if alarm else "STATUS: OK"
    color = (0, 0, 255) if alarm else (0, 255, 0)
    cv2.putText(frame, f"{status} | foreign_count={foreign_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 预览输出
    if view_mode == 0:
        show = frame
    elif view_mode == 1:
        show = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR)
    else:
        overlay = frame.copy()
        overlay[obj_mask > 0] = (0, 0, 255)  # 前景区域标红
        show = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # 在画面上显示当前阈值，方便你抄走固定下来
    info = f"Vmin={WHITE_V_MIN} Smax={WHITE_S_MAX} minA={MIN_AREA} K={K} open={OPEN_IT} close={CLOSE_IT} conf={CONF_TH:.2f} cover={COVER_TH:.2f}"
    cv2.putText(show, info, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("preview", show)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()