import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
import time
import os
import threading
from queue import Queue

app = Flask(__name__, static_folder='static')

VIDEO_FILENAME = 'axonfootage.mp4'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_count = 0
processing_time = 0
frame_queue = Queue(maxsize=10)

def process_video():
    global face_count, processing_time
    video = cv2.VideoCapture(os.path.join(app.static_folder, VIDEO_FILENAME))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps

    while True:
        start_time = time.time()
        success, frame = video.read()
        if not success:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_count = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        if not frame_queue.full():
            frame_queue.put(frame_bytes)

        processing_time = (time.time() - start_time) * 1000

        # Maintain original video frame rate
        time_to_wait = frame_delay - (time.time() - start_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analysis_data')
def analysis_data():
    global face_count, processing_time
    return jsonify({
        'faceCount': face_count,
        'processingTime': round(processing_time, 2)
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=process_video, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)