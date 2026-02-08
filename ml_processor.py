import cv2
import sqlite3
import datetime
import time
import threading
import yt_dlp
from flask import Flask, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('benguet_crowd.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS realtime_crowd 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  road_name TEXT, 
                  total_count INTEGER, 
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_data(road, count):
    conn = sqlite3.connect('benguet_crowd.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO realtime_crowd (road_name, total_count, timestamp) VALUES (?, ?, ?)",
              (road, count, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_latest_data():
    conn = sqlite3.connect('benguet_crowd.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""SELECT road_name, total_count FROM realtime_crowd 
                 WHERE id IN (SELECT MAX(id) FROM realtime_crowd GROUP BY road_name)""")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

# --- FLASK ROUTES ---
@app.route('/api/crowd')
def crowd_api():
    return jsonify(get_latest_data())

@app.route('/')
def index():
    return render_template('index.html')

# --- YOLO VISION LOGIC ---
def get_live_stream_url(webpage_url):
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(webpage_url, download=False)
        return info['url']

def process_live_camera(url, road_name):
    print(f"Connecting to stream: {road_name}...")
    try:
        stream_url = get_live_stream_url(url)
    except Exception as e:
        print(f"Error getting stream: {e}")
        return

    cap = cv2.VideoCapture(stream_url)
    model = YOLO('yolov8n.pt')
    
    # --- COUNTING CONFIG ---
    line_y = 400 # Horizontal line position
    counted_ids = set() # Stores IDs that already crossed this minute
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (1024, 576))
        
        # Track people (class 0)
        results = model.track(frame, persist=True, classes=[0], verbose=False, tracker="botsort.yaml")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) / 2 # Calculate center of person
                
                # --- LINE CROSSING LOGIC ---
                # If person crosses the line and hasn't been counted yet
                if center_y > line_y and obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                
                # Visuals: Box and ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y1 + 25)), (0, 255, 255), -1)
                cv2.putText(frame, f"ID:{obj_id}", (int(x1), int(y1)+18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        # --- DRAW DETECTION LINE ---
        cv2.line(frame, (0, line_y), (1024, line_y), (0, 0, 255), 3)
        cv2.putText(frame, "DETECTION LINE", (10, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # UI Overlay
        cv2.rectangle(frame, (0,0), (450, 70), (0,0,0), -1)
        cv2.putText(frame, f"FEED: {road_name}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"PASSED THIS MIN: {len(counted_ids)}", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Security Feed - Trigger Counter", frame)

        # LOG DATA EVERY 60 SECONDS
        if time.time() - start_time >= 60:
            print(f"MINUTE LOG: {len(counted_ids)} people crossed the line.")
            log_data(road_name, len(counted_ids))
            counted_ids.clear() # Reset for the new minute
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    init_db()
    
    # Soliman St (Agdao) YouTube Stream
    cam_url = "https://youtu.be/kCvjW21S_Dw" 
    cam_name = "Soliman St (Agdao)"

    vision_thread = threading.Thread(target=process_live_camera, args=(cam_url, cam_name))
    vision_thread.daemon = True
    vision_thread.start()

    app.run(debug=False, port=5000, use_reloader=False)