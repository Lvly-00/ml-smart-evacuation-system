import cv2
import sqlite3
import datetime
import time
from ultralytics import YOLO

# Initialize YOLOv8
model = YOLO('yolov8n.pt') 

def init_db():
    conn = sqlite3.connect('benguet_crowd.db')
    c = conn.cursor()
    # Stores the total cumulative count
    c.execute('''CREATE TABLE IF NOT EXISTS realtime_crowd 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  road_name TEXT, 
                  total_count INTEGER, 
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_data(road, total_count):
    conn = sqlite3.connect('benguet_crowd.db')
    c = conn.cursor()
    c.execute("INSERT INTO realtime_crowd (road_name, total_count, timestamp) VALUES (?, ?, ?)",
              (road, total_count, datetime.datetime.now()))
    conn.commit()
    conn.close()

def process_uploaded_video(video_path, road_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    line_y = 400 
    counted_ids = set() # Global set to ensure unique counting
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        frame = cv2.resize(frame, (1024, 576))

        # Track people (class 0)
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                # 1. Coordinate Logic
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) / 2
                
                # 2. "Head Area" Bounding Box Calculation
                # We take the top 25% of the person's detected box to simulate head detection
                head_height = (y2 - y1) * 0.25
                hx1, hy1, hx2, hy2 = int(x1), int(y1), int(x2), int(y1 + head_height)

                # 3. Visual: Draw Head Square & ID Number
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                cv2.putText(frame, f"P-{obj_id}", (hx1, hy1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 4. Counting Logic (Crosses the line)
                if center_y > line_y and obj_id not in counted_ids:
                    counted_ids.add(obj_id)

        # --- OVERLAY ---
        cv2.rectangle(frame, (0, 0), (1024, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"LT-CCTV: {road_name} | TOTAL PASSERBY: {len(counted_ids)}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Detection Line
        cv2.line(frame, (0, line_y), (1024, line_y), (0, 0, 255), 2)

        cv2.imshow("La Trinidad Security Monitor - Cumulative", frame)

        # UPDATE DB EVERY 10 SECONDS (No Reset)
        if time.time() - start_time >= 5:
            log_data(road_name, len(counted_ids))
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_db()
    # Make sure your file is named 'la_trinidad_feed.mp4'
    process_uploaded_video('la_trinidad_feed.mp4', "Halsema Hwy (Km. 5)")