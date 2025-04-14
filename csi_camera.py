from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import CircularOutput
import cv2

import os
import logging
from datetime import datetime
import utils

class CameraCSI():
    def __init__(self, device_name, video_wh=(1920,1080), model_wh=(640, 640), fps=5, use_bgr=False, is_pi5=False,
                 crop_to_square=False, calibration_file=None, save_video=False, buffer_secs=5,
                 create_preview=False, rotate_img="none", convert_h264=False):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Camera initialized!")

        self.device_name = device_name
        self.video_wh = video_wh
        self.model_wh = model_wh
        self.fps = fps
        self.use_bgr = use_bgr
        self.is_pi5 = is_pi5
        self.convert_h264 = convert_h264

        self.crop_to_square = crop_to_square
        self.calibration_file = calibration_file
        self.save_video = save_video
        self.buffer_secs = buffer_secs
        self.create_preview = create_preview
        self.rotate_img = rotate_img

        self.picam2 = Picamera2()

        # Configure camera streams
        # Hailo model needs BGR??
        if self.is_pi5:
            lores_format = 'BGR888' if self.use_bgr else 'RGB888'
        else:
            lores_format = "YUV420"
        self.logger.info(f"Low Res format {lores_format}")

        # Keep the aspect ratio of the main image in the lo-res image
        self.lores_wh = self.model_wh

        if self.crop_to_square:
            lores_w = int(round(self.model_wh[0] * (self.video_wh[0] / self.video_wh[1])))
            self.lores_wh = (lores_w, self.model_wh[1])

        logging.info(f"Low Res video shape HxW: {self.lores_wh[1]}, {self.lores_wh[0]}")
        lores = {'size': self.lores_wh, 'format': lores_format}

        main_res = {'size': self.video_wh, 'format': 'XRGB8888'}
        controls = {'FrameRate': self.fps}
        config = self.picam2.create_video_configuration(main_res, lores=lores, controls=controls)
        self.picam2.configure(config)

        # Create pre-callback to un-warp image with pre-calculated parameters
        if self.calibration_file:
            self.logger.info(f"Creating calibration params")
            cam_params = utils.get_calibration_params(self.calibration_file,
                                                      self.video_wh,
                                                      self.lores_wh)
            if cam_params:
                self.picam2.pre_callback = lambda req: utils.correct_image(req, cam_params)
            else:
                self.logger.warning(f"Could not create calibration params!")

        if self.create_preview:
            self.picam2.start_preview(Preview.QT, x=0, y=0, width=self.video_wh[0], height=self.video_wh[1])
            self.logger.info(f"Camera Creating Preview")

        self.picam2.start()

        if self.save_video:
            self.encoder = H264Encoder(1000000, repeat=True)
            self.output = CircularOutput(buffersize=self.buffer_secs * self.fps)
            self.picam2.start_recording(self.encoder, self.output, quality=Quality.HIGH)
            self.logger.info(f"Saving Video")

    def get_frames(self):
        # Capture and process frame
        (main_frame, frame), metadata = self.picam2.capture_arrays(["main", "lores"])

        if not self.is_pi5:
            if self.use_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)

        # Resize and crop to model size
        frame = utils.pre_process_image(frame, rotate=self.rotate_img,
                                        crop_to_square=self.crop_to_square)

        return main_frame, frame

    def start_video_recording(self, videos_detections_path):
        if self.save_video:
            self.logger.info("Starting Video recording!")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.join(videos_detections_path, f"{self.device_name}_{timestamp}.h264")
            self.output.fileoutput = file_name
            self.output.start()
        else:
            self.logger.info("Save video is not running!")

    def stop_video_recording(self):
        if self.save_video:
            self.logger.info("Stoping Video recording!")
            self.output.stop()
            if self.convert_h264:
                new_file_name = self.output.fileoutput.split(".")[0] + ".mp4"
                utils.convert_h264_to_mp4(self.output.fileoutput, new_file_name, self.fps)
        else:
            self.logger.info("Save video is not running!")

    def stop_camera(self):
        self.picam2.stop()
        self.picam2.close()

def main():
    camera = CameraCSI(video_wh=(1920,1080), model_wh=(640, 640), fps=30, create_preview=True)

    while True :
        _, _ = camera.get_frames()


if __name__ == "__main__":
    main()