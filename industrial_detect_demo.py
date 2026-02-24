import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# 1️⃣ 加载模型
# =============================
model = YOLO("./runs/detect/train9/weights/best.pt")   # 替换成你的模型路径

# =============================
# 2️⃣ 打开双摄像头
# =============================
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

# 可选：设置分辨率
for cam in [cam0, cam1]:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =============================
# 3️⃣ 主循环
# =============================
while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0 or not ret1:
        print("摄像头读取失败")
        break

    # =============================
    # 4️⃣ YOLO 推理
    # =============================
    results0 = model(frame0, conf=0.4)
    results1 = model(frame1, conf=0.4)

    # =============================
    # 5️⃣ 处理检测结果
    # =============================
    def process_results(frame, results):
        anomaly_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                # 坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                anomaly_count += 1

                # 绘制框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 标签
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        # 左上角显示数量
        cv2.putText(frame,
                    f"Anomaly Count: {anomaly_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if anomaly_count > 0 else (0, 255, 0),
                    2)

        # 显示 OK / NG
        status_text = "NG" if anomaly_count > 0 else "OK"
        color = (0, 0, 255) if anomaly_count > 0 else (0, 255, 0)

        cv2.putText(frame,
                    status_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    4)

        return frame, anomaly_count

    frame0, count0 = process_results(frame0, results0)
    frame1, count1 = process_results(frame1, results1)

    # =============================
    # 6️⃣ 拼接画面
    # =============================
    combined = np.vstack((frame0, frame1))

    cv2.imshow("Dual Camera YOLO Inspection", combined)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# 7️⃣ 释放资源
# =============================
cam0.release()
cam1.release()
cv2.destroyAllWindows()