import os
from flask import Flask, send_from_directory, render_template, jsonify, Response
import cv2
import numpy as np
import time

app = Flask(__name__, static_folder='static')

# Update this to your actual video file name
VIDEO_FILENAME = 'axonfootage.mp4'

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables to store analysis data
face_count = 0
processing_time = 0

def detect_faces(frame):
    global face_count, processing_time
    start_time = time.time()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_count = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return frame

@app.route('/video')
def serve_video():
    return send_from_directory(app.static_folder, VIDEO_FILENAME)

@app.route('/video_feed')
def video_feed():
    def generate():
        video = cv2.VideoCapture(os.path.join(app.static_folder, VIDEO_FILENAME))
        while True:
            success, frame = video.read()
            if not success:
                break
            else:
                frame = detect_faces(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        video.release()

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
    app_root = os.path.dirname(os.path.abspath(__file__))
    
    response = f"App root: {app_root}\n"
    response += f"Static folder path: {app.static_folder}\n"
    response += f"Looking for video at: {video_path}\n"
    response += f"Static folder contents: {os.listdir(app.static_folder)}\n"
    
    if os.path.exists(video_path):
        response += f"Video file exists. Size: {os.path.getsize(video_path)} bytes"
    else:
        response += "Video file not found"
    
    return response, 200 if os.path.exists(video_path) else 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)