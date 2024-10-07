import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_from_directory
import time
import os
import threading

app = Flask(__name__, static_folder='static')

# Update this to your actual video file name
VIDEO_FILENAME = 'axonfootage.mp4'

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables to store analysis data
face_count = 0
processing_time = 0
processed_frame = None

def process_video():
    global face_count, processing_time, processed_frame
    video = cv2.VideoCapture(os.path.join(app.static_folder, VIDEO_FILENAME))
    while True:
        success, frame = video.read()
        if not success:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue
        
        start_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_count = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        ret, buffer = cv2.imencode('.jpg', frame)
        processed_frame = buffer.tobytes()
        
        time.sleep(0.03)  # Adjust for desired frame rate

@app.route('/video')
def serve_video():
    return send_from_directory(app.static_folder, VIDEO_FILENAME)

@app.route('/video_feed')
def video_feed():
    def generate():
        global processed_frame
        while True:
            if processed_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')
            time.sleep(0.03)  # Adjust for desired frame rate

    return Response(generate(),
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

@app.route('/check_video')
def check_video():
    video_path = os.path.join(app.static_folder, VIDEO_FILENAME)
    if os.path.exists(video_path):
        return f"Video file exists. Size: {os.path.getsize(video_path)} bytes"
    else:
        return "Video file not found", 404

if __name__ == '__main__':
    threading.Thread(target=process_video, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)