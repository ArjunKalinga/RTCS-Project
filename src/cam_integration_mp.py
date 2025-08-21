import cv2
import mediapipe as mp
import time

# --- Configuration for Printing Keypoint Data ---
PRINT_INTERVAL_SECONDS = 1.0
last_print_time = time.time()

# --- MediaPipe Solution Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Please ensure your webcam is connected and not in use by another application.")
    exit()

print("Camera opened successfully. Displaying live feed. Press 'q' to quit.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # --- Main Loop for Video Processing ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting the video stream.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- Process and Draw Pose Landmarks on the Frame ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # --- Extract and Print Keypoint Data (Controlled Frequency) ---
            current_time = time.time()
            if current_time - last_print_time >= PRINT_INTERVAL_SECONDS:
                print("\n--- Keypoint Data (Updated) ---")

                # --- MODIFIED LANDMARK EXTRACTION START ---
                # Accessing different specific landmarks now: Mouth corners and Hips
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] # Keep nose for reference
                mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
                mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]


                # Print coordinates of the new set of landmarks
                print(f"Nose: (x={nose.x:.2f}, y={nose.y:.2f}, z={nose.z:.2f}, vis={nose.visibility:.2f})")
                print(f"L Mouth: (x={mouth_left.x:.2f}, y={mouth_left.y:.2f}, z={mouth_left.z:.2f}, vis={mouth_left.visibility:.2f})")
                print(f"R Mouth: (x={mouth_right.x:.2f}, y={mouth_right.y:.2f}, z={mouth_right.z:.2f}, vis={mouth_right.visibility:.2f})")
                print(f"L Hip: (x={left_hip.x:.2f}, y={left_hip.y:.2f}, z={left_hip.z:.2f}, vis={left_hip.visibility:.2f})")
                print(f"R Hip: (x={right_hip.x:.2f}, y={right_hip.y:.2f}, z={right_hip.z:.2f}, vis={right_hip.visibility:.2f})")
                print(f"L Knee: (x={left_knee.x:.2f}, y={left_knee.y:.2f}, z={left_knee.z:.2f}, vis={left_knee.visibility:.2f})")
                print(f"R Knee: (x={right_knee.x:.2f}, y={right_knee.y:.2f}, z={right_knee.z:.2f}, vis={right_knee.visibility:.2f})")
                # --- MODIFIED LANDMARK EXTRACTION END ---

                # This part to get ALL 33 landmarks remains the same and is very useful for your ML model
                all_landmarks_data = []
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    all_landmarks_data.append({
                        'id': id,
                        'name': mp_pose.PoseLandmark(id).name,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                # print("All Landmarks Data (first 3 landmarks):", all_landmarks_data[:3])

                last_print_time = current_time

        cv2.imshow('Live Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Program terminated. All resources released.")
