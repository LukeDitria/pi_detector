import logging
import cv2
import signal
import sys

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

    def get_frames(self):
        if self.cam.isOpened():
            ret, main_frame = self.cam.read()

            frame = cv2.resize(main_frame, self.lores_wh)

            # Resize and crop to model size
            frame = utils.pre_process_image(frame, rotate=self.rotate_img,
                                            crop_to_square=self.crop_to_square)

            if self.create_preview:
                cv2.imshow('frame', main_frame)

            return main_frame, frame
        else:
            return None, None

    def start_video_recording(self, filename):
        if self.save_video:
            self.logger.info("Save video is not running!")
        else:
            self.logger.info("Save video is not running!")

    def stop_video_recording(self):
        if self.save_video:
            self.logger.info("Save video is not running!")
        else:
            self.logger.info("Save video is not running!")

    def stop_camera(self):
        try:
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
    camera = CameraUSB(create_preview=True, crop_to_square=True)

    # Register signal handlers for safe termination
    camera_closer = lambda sig, frame: signal_handler(sig, frame, camera)
    signal.signal(signal.SIGINT, camera_closer)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, camera_closer)  # Handle `kill` command

    while(camera.cam.isOpened()):
        main_frame, frame = camera.get_frames()

        print(frame.shape)
        if main_frame is None:
            continue

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        # Is esc key?
        if k == 27:
            break

    camera.stop_camera()


if __name__ == "__main__":
    main()