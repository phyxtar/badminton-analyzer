from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import tempfile
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import threading
import logging

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose

# Logger setup for debugging
logging.basicConfig(level=logging.DEBUG)

def frame_to_base64(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_pose_and_analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS not detected

    best_shot_frame = None
    worst_shot_frame = None
    jump_frame = None

    movement_score = 0
    total_distance = 0
    time_seconds = 0

    hand_above_head_count = 0
    low_hand_count = 0
    side_movement_count = 0
    jump_count = 0

    prev_hip = None
    best_shot_conf = 0
    worst_shot_conf = 9999
    frame_index = 0

    best_frame_img = None
    worst_frame_img = None
    jump_frame_img = None

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to reduce processing load
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))  # Resize to 640x480

            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Best and worst shot logic
                if lm[16].y < lm[0].y:  # Right wrist above nose
                    hand_above_head_count += 1
                    height_diff = lm[0].y - lm[16].y
                    if height_diff > best_shot_conf:
                        best_shot_conf = height_diff
                        best_frame_img = frame.copy()
                else:
                    low_hand_count += 1
                    height_diff = lm[0].y - lm[16].y
                    if height_diff < worst_shot_conf:
                        worst_shot_conf = height_diff
                        worst_frame_img = frame.copy()

                # Movement logic
                hip = lm[24]
                if prev_hip:
                    dx = (hip.x - prev_hip.x)
                    dy = (hip.y - prev_hip.y)
                    dist = (dx**2 + dy**2)**0.5
                    movement_score += dist
                    total_distance += dist
                    if dist > 0.01:
                        side_movement_count += 1
                    if hip.y < prev_hip.y - 0.05:
                        jump_count += 1
                        jump_frame_img = frame.copy()
                prev_hip = hip

            frame_index += 1

    cap.release()

    total_frames = frame_index
    time_seconds = total_frames / fps if fps else 1
    speed = total_distance / time_seconds if time_seconds > 0 else 0  # normalized units/sec

    best_shot = "Smash" if hand_above_head_count > total_frames * 0.3 else "Clear"
    weakest_shot = "Drop" if low_hand_count > hand_above_head_count else "Flat Shot"
    movement_type = "Side Steps" if side_movement_count > jump_count else "Jump Movements"

    suggestions = []
    if movement_score < 0.5:
        suggestions.append("Move more aggressively across the court.")
    if hand_above_head_count < low_hand_count:
        suggestions.append("Try to hit higher shots with proper follow-through.")
    if jump_count < 5:
        suggestions.append("Incorporate more jumping smashes to add power.")
    if not suggestions:
        suggestions = ["Great form! Keep it up."]

    return {
        "shot": best_shot,
        "weakest_shot": weakest_shot,
        "movement_type": movement_type,
        "speed": round(speed, 3),  # Include speed
        "suggestions": suggestions,
        "images": {
            "best": frame_to_base64(best_frame_img) if best_frame_img is not None else None,
            "worst": frame_to_base64(worst_frame_img) if worst_frame_img is not None else None,
            "jump": frame_to_base64(jump_frame_img) if jump_frame_img is not None else None,
        }
    }

def analyze_video_thread(video_path):
    try:
        result = extract_pose_and_analyze(video_path)
        # Log the result or save it to a database, etc.
        logging.info("Analysis complete: %s", result)
    except Exception as e:
        logging.error(f"Error processing video: {e}")

@app.route('/')
def home():
    return send_file("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video'].read()

    # Check for file size
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    if len(video) > MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds the limit of 50 MB"}), 400

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
        temp_file.write(video)
        temp_file.flush()

        # Run the video analysis in a background thread
        threading.Thread(target=analyze_video_thread, args=(temp_file.name,)).start()

    return jsonify({"status": "Processing started. You will be notified when complete."})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # `use_reloader=False` to avoid double-threading during debug
