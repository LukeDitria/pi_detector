import logging
import cv2

class MotionDetector():
    def __init__(self, threshold=25, motion_percent=0.5):
        self.threshold = threshold
        self.motion_percent = motion_percent

        self.logger = logging.getLogger(__name__)
        self.previous_frame = None

    def detect_motion(self, current_frame):

        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(self.previous_frame, current_frame)

        # Apply threshold to identify areas with significant change
        _, motion_mask = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Optional: Apply noise reduction
        motion_mask = cv2.erode(motion_mask, None)
        motion_mask = cv2.dilate(motion_mask, None)

        # Calculate percentage of frame showing motion
        motion_percentage = (cv2.countNonZero(motion_mask) /
                             (motion_mask.shape[0] * motion_mask.shape[1])) * 100

        # Determine if meaningful motion is present (adjust this as needed)
        has_motion = motion_percentage > self.motion_percent

        return has_motion

    def get_detections(self, frame):
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame is not None:
            has_motion = self.detect_motion(current_frame)
        else:
            has_motion = False

        self.previous_frame = frame.copy()

        return has_motion