import logging
import cv2
import signal
import sys
import threading
import time
from queue import Queue
from datetime import datetime
import os

import utils

class CameraUSB():
    def __init__(self, video_wh=(1920,1080), model_wh=(640,640), fps=30, use_bgr=False,
                 crop_to_square=False, calibration_file=None, save_video=False, buffer_secs=5,
                 create_preview=False, rotate_img="none"):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Camera initialized!")

        self.video_wh = video_wh
        self.model_wh = model_wh
        self.fps = fps

        self.use_bgr = use_bgr
        self.crop_to_square = crop_to_square
        self.calibration_file = calibration_file
        self.save_video = save_video
        self.buffer_secs = buffer_secs
        self.create_preview = create_preview
        self.rotate_img = rotate_img

        self.cam = cv2.VideoCapture(0)
        # self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_wh[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_wh[1])

        self.lores_wh = self.model_wh
        if self.crop_to_square:
            lores_w = int(round(self.model_wh[0] * (self.video_wh[0] / self.video_wh[1])))
            self.lores_wh = (lores_w, self.model_wh[1])

        # Current frame storage
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.recording_lock = threading.Lock()

        # Control flag for the capture thread
        self.running = False
        self.capture_thread = None
        self.recording = False

        if self.save_video:
            self.buffer_size = self.buffer_secs * self.fps
            self.frame_buffer = Queue(maxsize=self.buffer_size)

        self.start()

    def start_video_recording(self, videos_detections_path):
        if self.save_video:
            with self.recording_lock:
                if self.recording:
                    self.logger.info("Already recording. Stop current recording first.")
                    return None

                # Generate filename if not provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.mp4"
                filepath = os.path.join(videos_detections_path, filename)

                # Create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, self.video_wh)
                self.recording = True

                # Launch a separate thread to write buffer frames
                buffer_writer_thread = threading.Thread(
                    target=self._write_buffer_frames,
                    daemon=True
                )
                buffer_writer_thread.start()
        else:
            self.logger.info(f"Not recording! save_video is False!")

    def _write_buffer_frames(self):
        while True:
            if self.video_writer is not None:
                if not self.frame_buffer.empty():
                    frame = self.frame_buffer.get()
                    self.video_writer.write(frame)
            else:
                break

    def stop_video_recording(self):
        if self.recording:
            with self.recording_lock:
                self.recording = False
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                    self.logger.info("Stopping recording")
        else:
            self.logger.info(f"Not recording! save_video is False!")

    def stop(self):
        """Stop the camera capture thread."""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

    def start(self):
        """Start the camera capture thread."""
        if self.running:
            self.logger.info("Camera capture is already running.")
            return

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Allow time for camera to initialize
        time.sleep(0.5)

        return self

    def _capture_loop(self):
        if not self.cam.isOpened():
            self.running = False
            return

        try:
            while self.running:
                ret, frame = self.cam.read()

                if not ret:
                    self.logger.warning("Warning: Failed to capture frame")
                    time.sleep(0.1)
                    continue

                with self.frame_lock:
                    self.current_frame = frame

                if self.save_video:
                    # Add to circular buffer
                    if self.frame_buffer.full():
                        _ = self.frame_buffer.get()
                    self.frame_buffer.put(frame.copy(), block=True, timeout=0.01)

                    # Handle recording
                    if self.recording:
                        with self.recording_lock:
                            self.video_writer.write(frame)

        finally:
            # Release the camera when done
            self.cam.release()

    def get_frames(self):

        with self.frame_lock:
            if self.current_frame is None:
                return None, None
            main_frame = self.current_frame.copy()

        frame = cv2.resize(main_frame, self.lores_wh)

        # Resize and crop to model size
        frame = utils.pre_process_image(frame, rotate=self.rotate_img,
                                        crop_to_square=self.crop_to_square)

        if self.create_preview:
            cv2.imshow('frame', main_frame)

        if self.use_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return main_frame, frame

    def stop_camera(self):
        try:
            self.stop()
            self.cam.release()
        except Exception as e:
            self.logger.warning("Could not close camera! {e}")

def signal_handler(sig, frame, cam):
    """Handle SIGINT (Ctrl+C) and SIGTERM (kill command) to release camera."""
    print("\nCaught signal, releasing camera...")
    cam.stop_camera()
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    camera = CameraUSB(create_preview=True, crop_to_square=True, save_video=True)

    # Register signal handlers for safe termination
    camera_closer = lambda sig, frame: signal_handler(sig, frame, camera)
    signal.signal(signal.SIGINT, camera_closer)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, camera_closer)  # Handle `kill` command

    while(camera.cam.isOpened()):
        main_frame, frame = camera.get_frames()

        if main_frame is None:
            continue

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)

        if key != -1:
            print(f"got Key {key}")

        # Is esc key?
        if key == ord('q'):
            break

        # Process key commands
        if key == ord('r') and not camera.recording:
            print("Start Recording!")
            # Start recording including buffer
            camera.start_video_recording(".")

        elif key == ord('s') and camera.recording:
            print("Stop Recording!")
            # Stop recording
            camera.stop_video_recording()

    camera.stop_camera()


if __name__ == "__main__":
    main()