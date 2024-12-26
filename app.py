from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from utils.yolo_predictions import YOLO_Pred
from utils.count_yolo_predictions import count_YOLO_Pred
import subprocess

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'uploads/results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# YOLO models
single_chicken_yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')
group_chicken_yolo = count_YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    task = request.form['task']

    if task == 'single_chicken':
        image = cv2.imread(file_path)
        result = single_chicken_yolo.predictions(image)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'single_' + filename)
        cv2.imwrite(result_path, result)
        return render_template('results.html', result_image='single_' + filename, details=None)

    elif task == 'group_chickens':
        image = cv2.imread(file_path)
        result, counts = group_chicken_yolo.predictions(image)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'group_' + filename)
        cv2.imwrite(result_path, result)
        count_details = {
            "Healthy": counts.get("Healthy", 0),
            "Unhealthy": counts.get("Unhealthy", 0),
            "Not_a_Chicken": counts.get("Not_a_Chicken", 0)
        }
        return render_template('results.html', result_image='group_' + filename, details=count_details)

    return "Task not recognized"

@app.route('/process_video', methods=['POST'])
def process_video():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Define output path and format
    result_filename = 'processed_' + os.path.splitext(filename)[0] + '.mp4'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pred_frame = single_chicken_yolo.predictions(frame)
        out.write(pred_frame)

    cap.release()
    out.release()

    # Open the processed video in VLC and wait for it to close
    try:
        vlc_path = "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe"  # Path to VLC executable
        vlc_process = subprocess.Popen([vlc_path, result_path])  # Open the video in VLC
        vlc_process.wait()  # Wait for VLC to close
    except FileNotFoundError:
        try:
            vlc_path = "C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe"  # Path to VLC executable
            vlc_process = subprocess.Popen([vlc_path, result_path])  # Open the video in VLC
            vlc_process.wait()  # Wait for VLC to close
        except:
            return "VLC Media Player not found. Please check your installation or path.", 500

    # Redirect to the home page after VLC is closed
    return render_template('index.html')  # Render the home page

@app.route('/uploads/results/<filename>')
def send_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)



if __name__ == '__main__':
    app.run(debug=True)
