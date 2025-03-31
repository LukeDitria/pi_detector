import logging
import cv2

import utils

class CameraUSB():
    def __init__(self, model_wh=(640, 640), use_bgr=False,
                 crop_to_square=False, calibration_file=None, save_video=False, buffer_secs=5,
                 create_preview=False, rotate_img="none"):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Camera initialized!")

        self.model_wh = model_wh
        self.use_bgr = use_bgr
        self.crop_to_square = crop_to_square
        self.calibration_file = calibration_file
        self.save_video = save_video
        self.buffer_secs = buffer_secs
        self.create_preview = create_preview
        self.rotate_img = rotate_img

        self.cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
        # self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_wh[0])
        # self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_wh[1])
        self.video_wh = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.fps = int(self.cam.get(cv2.CAP_PROP_FPS))

        if self.crop_to_square:
            lores_w = int(round(self.model_wh[0] * (self.video_wh[0] / self.video_wh[1])))
            self.lores_wh = (lores_w, self.model_wh[1])

    def get_frames(self):
        # Capture and process frame
        ret, main_frame = self.cam.read()

        frame = cv2.resize(main_frame, self.lores_wh)

        # Resize and crop to model size
        frame = utils.pre_process_image(frame, rotate=self.rotate_img,
                                        crop_to_square=self.crop_to_square)

        if self.create_preview:
            cv2.imshow('frame', main_frame)

        return main_frame, frame

    # def start_video_recording(self, filename):
    #     if self.save_video:
    #
    #     else:
    #         self.logger.info("Save video is not running!")
    #
    # def stop_video_recording(self):
    #     if self.save_video:
    #     else:
    #         self.logger.info("Save video is not running!")

    def stop_camera(self):
        self.cam.release()

def main():
    camera = CameraUSB(create_preview=True)

    while True :
        _, _ = camera.get_frames()


if __name__ == "__main__":
    main()