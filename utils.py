import numpy as np
import cv2
import os
import pickle
from picamera2 import Picamera2
from picamera2 import MappedArray, Preview
import subprocess


def read_class_list(filepath):
    """Read list of class names from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def extract_detections(hailo_output, class_names, valid_classes, threshold=0.5, hailo_aspect=1):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        class_name = class_names[class_id]
        if valid_classes and class_name not in valid_classes:
            continue

        for detection in detections:
            y0, x0, y1, x1 = detection[:4]
            bbox = (float(x0) / hailo_aspect, float(y0), float(x1) / hailo_aspect, float(y1))
            score = detection[4]
            if score >= threshold:
                results.append([class_name, bbox, score])
    return results


def log_detection(filename, output_dir, detections):
    """Log detection results to a JSON file."""
    import json
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    results = {
        "timestamp": timestamp,
        "filename": filename,
        "detections": [
            {
                "class": class_name,
                "confidence": float(score),
                "bbox": bbox
            }
            for class_name, bbox, score in detections
        ]
    }

    json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

def pre_process_image(image, rotate="cw", crop_to_square=False):
    if rotate == "cw":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "ccw":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == "flip":
        image = cv2.rotate(image, cv2.ROTATE_180)

    if crop_to_square:
        # assuming that w > h
        h, w, _ = image.shape
        if not h == w:
            split = (w - h)//2
            image = np.ascontiguousarray(image[:, split:split+h])

    return image

def find_first_usb_drive():
    # Relies on raspi OS to auto mount USB storage to /media/username etc
    media_path = "/media"

    # Check if /media exists
    if not os.path.exists(media_path):
        return None

    # Get all user directories under /media (usually just one)
    media_items = os.listdir(media_path)

    for user_dir in media_items:
        user_path = os.path.join(media_path, user_dir)

        # If this is a directory
        if os.path.isdir(user_path):
            # Check for any subdirectories (mounted drives)
            try:
                usb_drives = os.listdir(user_path)
                if usb_drives:
                    # Return the first drive found
                    return os.path.join(user_path, usb_drives[0])
            except:
                pass

    # No USB drives found
    return None

def correct_image(request, cam_params):

    with MappedArray(request, "lores") as m:
        x, y, w, h = cam_params['lores_roi']
        undistorted = cv2.undistort(m.array, cam_params['lores_mtx'], cam_params['lores_dist'],
                                    None, cam_params['lores_newcameramtx'])
        undistorted = undistorted[y:y + h, x:x + w]
        undistorted = cv2.resize(undistorted, (m.array.shape[1], m.array.shape[0]))
        np.copyto(m.array, undistorted)

    with MappedArray(request, "main") as m:
        x, y, w, h = cam_params['main_roi']
        undistorted = cv2.undistort(m.array, cam_params['main_mtx'], cam_params['main_dist'],
                                    None, cam_params['main_newcameramtx'])
        undistorted = undistorted[y:y + h, x:x + w]
        undistorted = cv2.resize(undistorted, (m.array.shape[1], m.array.shape[0]))
        np.copyto(m.array, undistorted)


def get_calibration_params(calibration_file, main_wh, lores_wh):
    if os.path.exists(calibration_file):
        with open(calibration_file, 'rb') as f:
            calibration_data = pickle.load(f)

        main_mtx = calibration_data['main_camera_matrix']
        main_dist = calibration_data['main_dist_coeffs']
        lores_mtx = calibration_data['lores_camera_matrix']
        lores_dist = calibration_data['lores_dist_coeffs']

        main_newcameramtx, main_roi = cv2.getOptimalNewCameraMatrix(main_mtx, main_dist,
                                                                    main_wh, 1,
                                                                    main_wh)

        lores_newcameramtx, lores_roi = cv2.getOptimalNewCameraMatrix(lores_mtx, lores_dist,
                                                                      lores_wh, 1,
                                                                      lores_wh)

        cam_params = {"main_mtx": main_mtx,
                      "main_dist": main_dist,
                      "main_newcameramtx": main_newcameramtx,
                      "main_roi": main_roi,
                      "lores_mtx": lores_mtx,
                      "lores_dist": lores_dist,
                      "lores_newcameramtx": lores_newcameramtx,
                      "lores_roi": lores_roi
                      }

        return cam_params
    else:
        return None

def parse_resolution(image_size):
    # Parse video size
    if isinstance(image_size, str):
        video_w, video_h = map(int, image_size.split(','))
    else:
        # Handle case where video_size might be a list/tuple in the JSON
        video_w, video_h = image_size

    return video_w, video_h

def convert_h264_to_mp4(input_path, output_path, framerate=30):
    command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_path,
        '-c', 'copy',
        output_path
    ]
    try:
        subprocess.Popen(command, check=True)
        print(f"Converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")