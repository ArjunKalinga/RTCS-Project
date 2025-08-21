import cv2
import numpy as np
import time

# --- Dashboard & Alert Configuration ---
ALERT_ENGAGEMENT_THRESHOLD = 60.0 # Alert if engagement is below 60%
ALERT_TIME_THRESHOLD_SECS = 60.0 # For 10 consecutive seconds

# --- Alert State Variables ---
# We track the timer outside the function to maintain state across frames
low_engagement_start_time = None
alert_active = False

def draw_dashboard(frame, stats):
    """
    Draws the summary dashboard and handles the visual alert system.

    Args:
        frame: The video frame to draw on.
        stats: A dictionary containing the student counts.

    Returns:
        The frame with the dashboard drawn on it.
    """
    global low_engagement_start_time, alert_active
    
    # Extract stats from the dictionary
    total_students = stats.get("total_students", 0)
    attentive_count = stats.get("attentive_count", 0)
    inattentive_count = stats.get("inattentive_count", 0)
    writing_count = stats.get("writing_count", 0)
    sleeping_count = stats.get("sleeping_count", 0)

    # Calculate overall attentiveness percentage
    total_attentive = attentive_count + writing_count
    attentiveness_percentage = (total_attentive / total_students) * 100 if total_students > 0 else 0

    # --- Visual Alert Logic ---
    current_time = time.time()
    if attentiveness_percentage < ALERT_ENGAGEMENT_THRESHOLD and total_students > 0:
        if low_engagement_start_time is None:
            # If engagement just dropped, start the timer
            low_engagement_start_time = current_time
        else:
            # If timer is running, check if it has exceeded the threshold
            if current_time - low_engagement_start_time > ALERT_TIME_THRESHOLD_SECS:
                alert_active = True
    else:
        # If engagement is good, reset the timer and turn off the alert
        low_engagement_start_time = None
        alert_active = False

    # Determine panel color based on the alert status
    panel_color = (0, 0, 128) if alert_active else (0, 0, 0) # Dark red if alert, black otherwise

    # Create a semi-transparent panel for the dashboard
    overlay = frame.copy()
    panel_x, panel_y, panel_w, panel_h = 20, 20, 350, 210
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), panel_color, -1)
    alpha = 0.6 # Transparency factor
    annotated_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display the summary text on the panel
    cv2.putText(annotated_frame, f"Total Students: {total_students}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Attentive: {attentive_count}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Writing: {writing_count}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Inattentive: {inattentive_count}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Sleeping: {sleeping_count}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)
    cv2.putText(annotated_frame, f"Engagement: {attentiveness_percentage:.1f}%", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return annotated_frame
