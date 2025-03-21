import numpy as np
import cv2
import os
import pickle
import time
from picamera2 import Picamera2
from libcamera import Transform
from picamera2 import MappedArray, Preview


def camera_calibration():
    # Parameters
    board_size = (9, 6)  # Number of inner corners in the chessboard pattern
    square_size = 1.0  # Size in arbitrary units
    num_images_needed = 15  # Number of calibration images to capture
    calibration_file = "camera_calibration.pkl"

    # Initialize Picamera2
    print("Initializing camera...")
    picam2 = Picamera2()

    # Configure the camera
    preview_config = picam2.create_preview_configuration(
        main={"size": (1920,1080), "format": "RGB888"},
        lores={'size': (1280, 640), "format": "RGB888"})
    picam2.configure(preview_config)

    # Start the camera
    picam2.start()

    # Allow camera to warm up
    time.sleep(2)

    # Capture a frame to get the resolution
    (main_frame, lores_frame), metadata = picam2.capture_arrays(["main", "lores"])

    main_img_shape = (main_frame.shape[1], main_frame.shape[0])  # (width, height)
    lores_img_shape = (lores_frame.shape[1], lores_frame.shape[0])  # (width, height)

    print(f"Main Camera resolution: {main_img_shape[0]} x {main_img_shape[1]}")

    # Prepare object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points
    main_objpoints = []  # 3D points in real world space
    main_imgpoints = []  # 2D points in image plane

    lores_objpoints = []  # 3D points in real world space
    lores_imgpoints = []  # 2D points in image plane

    # Capture images for calibration
    images_captured = 0
    last_capture_time = time.time() - 5  # Allow immediate first capture

    print("\n==== CAMERA CALIBRATION CAPTURE ====")
    print(f"Hold a {board_size[0]}x{board_size[1]} chessboard pattern in front of the camera.")
    print("The program will automatically capture when a valid chessboard is detected.")
    print("Try to capture the pattern from different angles and distances.")
    print("Press 'ESC' to exit early or 'c' to force capture.")
    print(f"Need to capture {num_images_needed} images.")

    while images_captured < num_images_needed:
        # Capture frame
        (main_frame, lores_frame), metadata = picam2.capture_arrays(["main", "lores"])

        # Create a copy of the frame for drawing
        display_frame = main_frame.copy()

        # Convert to grayscale - picam2 returns RGB888, so we convert differently than with OpenCV
        lores_gray = cv2.cvtColor(lores_frame, cv2.COLOR_RGB2GRAY)
        main_gray = cv2.cvtColor(main_frame, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        lores_ret, lores_corners = cv2.findChessboardCorners(lores_gray, board_size, None)
        main_ret, main_corners = cv2.findChessboardCorners(main_gray, board_size, None)

        # Add text showing progress
        cv2.putText(display_frame, f"Captured: {images_captured}/{num_images_needed}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        current_time = time.time()
        time_since_last = current_time - last_capture_time

        # If corners are found, draw them and consider capturing
        if lores_ret and main_ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            lores_corners2 = cv2.cornerSubPix(lores_gray, lores_corners, (11, 11), (-1, -1), criteria)
            main_corners2 = cv2.cornerSubPix(main_gray, main_corners, (11, 11), (-1, -1), criteria)

            # Draw the corners
            cv2.drawChessboardCorners(display_frame, board_size, main_corners2, main_ret)

            # Auto-capture if enough time has passed (to ensure diverse images)
            force_capture = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                force_capture = True

            if force_capture or time_since_last > 2:  # Wait at least 2 seconds between captures
                main_objpoints.append(objp)
                main_imgpoints.append(main_corners2)

                lores_objpoints.append(objp)
                lores_imgpoints.append(lores_corners2)

                images_captured += 1
                last_capture_time = current_time

                # Display info about the capture
                print(f"Image {images_captured}/{num_images_needed} captured!")

                # Add a visual feedback for the capture
                cv2.putText(display_frame, "CAPTURED!", (display_frame.shape[1] // 2 - 100, display_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow('Camera Calibration', display_frame)
                cv2.waitKey(500)  # Pause briefly to show the capture message

        # Display the frame
        cv2.imshow('Camera Calibration', display_frame)

        # Exit if ESC is pressed
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            if images_captured > 5:  # Allow exit if we have at least 5 images
                print(f"Exiting early with {images_captured} images. This should be enough for basic calibration.")
                break
            else:
                print(f"Need at least 5 images for calibration. Currently have {images_captured}.")

    # Stop the camera before calibration
    picam2.stop()
    picam2.close()

    cv2.destroyAllWindows()

    if images_captured == 0:
        print("No calibration images captured. Exiting.")
        return

    # Perform camera calibration
    print("\nCalculating calibration parameters...")
    main_ret, main_mtx, main_dist, _, _ = cv2.calibrateCamera(
        main_objpoints, main_imgpoints, main_img_shape, None, None
    )

    lores_ret, lores_mtx, lores_dist, _, _ = cv2.calibrateCamera(
        lores_objpoints, lores_imgpoints, lores_img_shape, None, None
    )

    if main_ret and lores_ret:
        print(f"Calibration successful with RMS errors: {main_ret}, {lores_ret}")
        # Save calibration parameters
        calibration_data = {
            'main_camera_matrix': main_mtx,
            'main_dist_coeffs': main_dist,
            'lores_camera_matrix': lores_mtx,
            'lores_dist_coeffs': lores_dist,
        }

        with open(calibration_file, 'wb') as f:
            pickle.dump(calibration_data, f)

        print(f"Calibration parameters saved to {calibration_file}")

        # Test the calibration on a live feed
        test_calibration(calibration_file)
    else:
        print("Calibration failed.")

def correct_image(request, main_mtx, main_dist, main_newcameramtx, main_roi,
                  lores_mtx, lores_dist, lores_newcameramtx, lores_roi):

    with MappedArray(request, "lores") as m:
        x, y, w, h = lores_roi
        undistorted = cv2.undistort(m.array, lores_mtx, lores_dist, None, lores_newcameramtx)
        undistorted = undistorted[y:y + h, x:x + w]
        undistorted = cv2.resize(undistorted, (m.array.shape[1], m.array.shape[0]))
        np.copyto(m.array, undistorted)

    with MappedArray(request, "main") as m:
        x, y, w, h = main_roi
        undistorted = cv2.undistort(m.array, main_mtx, main_dist, None, main_newcameramtx)
        undistorted = undistorted[y:y + h, x:x + w]
        undistorted = cv2.resize(undistorted, (m.array.shape[1], m.array.shape[0]))
        np.copyto(m.array, undistorted)

def test_calibration(calibration_file):
    print("\n==== TESTING CALIBRATION ====")

    # Load existing calibration data
    with open(calibration_file, 'rb') as f:
        calibration_data = pickle.load(f)

    main_mtx = calibration_data['main_camera_matrix']
    main_dist = calibration_data['main_dist_coeffs']
    lores_mtx = calibration_data['lores_camera_matrix']
    lores_dist = calibration_data['lores_dist_coeffs']

    # Initialize Picamera2 again for testing
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (1920,1080), "format": "RGB888"},
        lores={'size': (1280, 640), "format": "RGB888"})
    picam2.configure(preview_config)

    main_newcameramtx, main_roi = cv2.getOptimalNewCameraMatrix(main_mtx, main_dist,
                                                                (1920,1080), 1,
                                                                (1920,1080))

    lores_newcameramtx, lores_roi = cv2.getOptimalNewCameraMatrix(lores_mtx, lores_dist,
                                                                (1280, 640), 1,
                                                                (1280, 640))

    picam2.pre_callback = lambda req: correct_image(req, main_mtx, main_dist, main_newcameramtx, main_roi,
                                                    lores_mtx, lores_dist, lores_newcameramtx, lores_roi)

    picam2.start()

    # Allow camera to warm up
    time.sleep(1)
    while True:
        # Capture frame
        (main_frame, frame), metadata = picam2.capture_arrays(["main", "lores"])

        # Add labels
        cv2.putText(frame, "Corrected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Calibration Test Corrected', frame)

        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()


def main():
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console
    print("===== PI CAMERA CALIBRATION TOOL =====")

    # Check if calibration file exists
    calibration_file = "camera_calibration.pkl"
    if os.path.exists(calibration_file):
        print(f"Found existing calibration file: {calibration_file}")
        print("Options:")
        print("1. Perform new calibration")
        print("2. Test existing calibration")
        choice = input("Enter your choice (1/2): ")

        if choice == "2":
            # Test with existing calibration
            test_calibration(calibration_file)
            return

    # Perform new calibration
    camera_calibration()
    print("\nCalibration process completed.")


if __name__ == "__main__":
    main()