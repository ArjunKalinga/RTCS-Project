import cv2
import mediapipe as mp

class PoseDetector:
    """
    A class to detect and draw human poses using MediaPipe.
    """
    def __init__(self, mode=False, complexity=0, smooth=True,
                 enable_seg=False, smooth_seg=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the PoseDetector with MediaPipe pose solution.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=mode,
                                      model_complexity=complexity,
                                      smooth_landmarks=smooth,
                                      enable_segmentation=enable_seg,
                                      smooth_segmentation=smooth_seg,
                                      min_detection_confidence=detectionCon,
                                      min_tracking_confidence=trackCon)
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        """
        Processes an image to find the pose landmarks.

        Args:
            img: The image to process.
            draw: Flag to draw the landmarks on the image.

        Returns:
            The image with the landmarks drawn on it.
        """
        # Convert the BGR image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and find landmarks
        self.results = self.pose.process(img_rgb)

        # Draw the landmarks on the image if draw is True and landmarks are found
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        """
        Finds the coordinates of each landmark. (We are not using this yet)

        Args:
            img: The image to process.
            draw: Flag to draw circles on the landmarks.

        Returns:
            A list of landmark coordinates.
        """
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list