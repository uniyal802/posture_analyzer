from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import datetime
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180: angle = 360 - angle
    return angle

def analyze_posture(image, mode="sitting"):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return {
            "bad_posture": True,
            "message": "No person detected",
            "score": 0,
            "landmarks": []
        }
    lm = results.pose_landmarks.landmark
    landmarks = [{"x": lmk.x, "y": lmk.y} for lmk in lm]

    # Example: Desk Sitting logic
    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    ear = [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x, lm[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    neck_angle = calculate_angle(shoulder, hip, ear)
    back_angle = calculate_angle(shoulder, hip, [hip[0], hip[1]+0.1])

    if neck_angle > 30:
        return {
            "bad_posture": True,
            "message": f"Neck bent: {int(neck_angle)}°",
            "score": 30,
            "landmarks": landmarks
        }
    if back_angle < 170:
        return {
            "bad_posture": True,
            "message": f"Back not straight: {int(back_angle)}°",
            "score": 40,
            "landmarks": landmarks
        }
    return {
        "bad_posture": False,
        "message": f"Posture analyzed successfully for mode: {mode}",
        "score": 100,
        "landmarks": landmarks
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.args.get("mode", "sitting")

    if "frame" not in request.files:
        return jsonify({"bad_posture": True, "message": "No frame uploaded", "score": 0, "landmarks": []}), 400

    frame = request.files["frame"]

    if frame.filename == "":
        return jsonify({"bad_posture": True, "message": "No selected file", "score": 0, "landmarks": []}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{mode}_{timestamp}_{frame.filename}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    frame.save(save_path)

    image = cv2.imread(save_path)
    if image is None:
        return jsonify({
            "bad_posture": True,
            "message": "Invalid image",
            "score": 0,
            "landmarks": []
        }), 400

    result = analyze_posture(image, mode)
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
