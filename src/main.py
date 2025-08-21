import cv2
from ultralytics import YOLO
import tkinter as tk
import numpy as np
import time
from dashboard import draw_dashboard
import csv
import os
from datetime import datetime

# Load the YOLOv8-Pose model
model = YOLO('yolov8n-pose.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

print("Starting real-time classification. Press 'q' to quit and save session.")

# --- Fullscreen Setup ---
window_name = "Classroom Attentiveness Classification"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get screen resolution
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# --- Keypoint IDs & State Tracking ---
NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 5, 6, 9, 10, 11, 12
CONF_THRESHOLD = 0.5
SLEEP_THRESHOLD_SECS = 120.0

student_states = {} # {track_id: [status, timestamp]}

# --- Session Logging Setup ---
LOG_DIRECTORY = 'data'
LOG_FILE = os.path.join(LOG_DIRECTORY, 'session_log.csv')
os.makedirs(LOG_DIRECTORY, exist_ok=True) # Creates the 'data' folder if it doesn't exist

session_start_time = time.time()
session_stats_agg = {
    "attentive": 0, "inattentive": 0, "writing": 0, "sleeping": 0
}
session_total_frames = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    stats = {
        "total_students": 0, "attentive_count": 0, "inattentive_count": 0,
        "writing_count": 0, "sleeping_count": 0
    }

    # --- CLASSIFICATION LOGIC ---
    if results[0].boxes.id is not None:
        keypoints_data = results[0].keypoints.data
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        stats["total_students"] = len(track_ids)

        for i, track_id in enumerate(track_ids):
            person_keypoints = keypoints_data[i]
            status = "INATTENTIVE"

            nose_y, nose_conf = person_keypoints[NOSE][1], person_keypoints[NOSE][2]
            l_sh_y, l_sh_conf = person_keypoints[L_SHOULDER][1], person_keypoints[L_SHOULDER][2]
            r_sh_y, r_sh_conf = person_keypoints[R_SHOULDER][1], person_keypoints[R_SHOULDER][2]
            l_wrist_y, l_wrist_conf = person_keypoints[L_WRIST][1], person_keypoints[L_WRIST][2]
            r_wrist_y, r_wrist_conf = person_keypoints[R_WRIST][1], person_keypoints[R_WRIST][2]
            l_hip_y, l_hip_conf = person_keypoints[L_HIP][1], person_keypoints[L_HIP][2]
            r_hip_y, r_hip_conf = person_keypoints[R_HIP][1], person_keypoints[R_HIP][2]
            
            if nose_conf > CONF_THRESHOLD and l_sh_conf > CONF_THRESHOLD and r_sh_conf > CONF_THRESHOLD:
                shoulder_avg_y = (l_sh_y + r_sh_y) / 2
                if nose_y < shoulder_avg_y - 10:
                    status = "ATTENTIVE"
                else:
                    if l_hip_conf > CONF_THRESHOLD and r_hip_conf > CONF_THRESHOLD and (l_wrist_conf > CONF_THRESHOLD or r_wrist_conf > CONF_THRESHOLD):
                        hip_avg_y = (l_hip_y + r_hip_y) / 2
                        left_wrist_in_area = shoulder_avg_y < l_wrist_y < hip_avg_y
                        right_wrist_in_area = shoulder_avg_y < r_wrist_y < hip_avg_y
                        if left_wrist_in_area or right_wrist_in_area:
                            status = "WRITING"
            
            current_time = time.time()
            if status == "INATTENTIVE":
                if track_id in student_states and student_states[track_id][0] == "INATTENTIVE":
                    if current_time - student_states[track_id][1] > SLEEP_THRESHOLD_SECS:
                        status = "SLEEPING"
                else:
                    student_states[track_id] = ["INATTENTIVE", current_time]
            else:
                student_states[track_id] = [status, current_time]

            if status == "ATTENTIVE": stats["attentive_count"] += 1
            elif status == "WRITING": stats["writing_count"] += 1
            elif status == "SLEEPING": stats["sleeping_count"] += 1
            else: stats["inattentive_count"] += 1

        session_stats_agg["attentive"] += stats["attentive_count"]
        session_stats_agg["inattentive"] += stats["inattentive_count"]
        session_stats_agg["writing"] += stats["writing_count"]
        session_stats_agg["sleeping"] += stats["sleeping_count"]
        session_total_frames += 1

    annotated_frame = draw_dashboard(annotated_frame, stats)
    display_frame = cv2.resize(annotated_frame, (screen_width, screen_height))
    cv2.imshow(window_name, display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Session Summary & Logging ---
session_end_time = time.time()
session_duration_secs = session_end_time - session_start_time

if session_total_frames > 0:
    total_classifications = sum(session_stats_agg.values())
    if total_classifications > 0:
        avg_attentive_pct = (session_stats_agg["attentive"] / total_classifications) * 100
        avg_inattentive_pct = (session_stats_agg["inattentive"] / total_classifications) * 100
        avg_writing_pct = (session_stats_agg["writing"] / total_classifications) * 100
        avg_sleeping_pct = (session_stats_agg["sleeping"] / total_classifications) * 100
        total_engaged_classifications = session_stats_agg["attentive"] + session_stats_agg["writing"]
        avg_overall_engagement_pct = (total_engaged_classifications / total_classifications) * 100

        log_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration (s)": f"{session_duration_secs:.1f}",
            "Avg Engagement (%)": f"{avg_overall_engagement_pct:.1f}",
            "Attentive (%)": f"{avg_attentive_pct:.1f}",
            "Writing (%)": f"{avg_writing_pct:.1f}",
            "Inattentive (%)": f"{avg_inattentive_pct:.1f}",
            "Sleeping (%)": f"{avg_sleeping_pct:.1f}"
        }

        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)
        
        print(f"Session summary saved to {LOG_FILE}")

print("Stopping program.")
cap.release()
cv2.destroyAllWindows()
