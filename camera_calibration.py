import numpy as np
import cv2
import os
import pickle
import time
from picamera2 import Picamera2
from libcamera import Transform


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
        main={"size": (1280, 720), "format": "RGB888"},
        transform=Transform(hflip=True, vflip=True)  # Flip if needed, adjust based on your camera orientation
    )
    picam2.configure(preview_config)

    # Start the camera
    picam2.start()

    # Allow camera to warm up
    time.sleep(2)

    # Capture a frame to get the resolution
    frame = picam2.capture_array()
    img_shape = (frame.shape[1], frame.shape[0])  # (width, height)
    print(f"Camera resolution: {img_shape[0]} x {img_shape[1]}")

    # Prepare object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

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
        frame = picam2.capture_array()

        # Create a copy of the frame for drawing
        display_frame = frame.copy()

        # Convert to grayscale - picam2 returns RGB888, so we convert differently than with OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        # Add text showing progress
        cv2.putText(display_frame, f"Captured: {images_captured}/{num_images_needed}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        current_time = time.time()
        time_since_last = current_time - last_capture_time

        # If corners are found, draw them and consider capturing
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw the corners
            cv2.drawChessboardCorners(display_frame, board_size, corners2, ret)

            # Auto-capture if enough time has passed (to ensure diverse images)
            force_capture = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                force_capture = True

            if force_capture or time_since_last > 2:  # Wait at least 2 seconds between captures
                objpoints.append(objp)
                imgpoints.append(corners2)
                images_captured += 1
                last_capture_time = current_time

                # Display info about the capture
                print(f"Image {images_captured}/{num_images_needed} captured!")

                # Add a visual feedback for the capture
                cv2.putText(display_frame, "CAPTURED!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow('Camera Calibration', display_frame)
                cv2.waitKey(500)  # Pause briefly to show the capture message

        # Display the frame - convert from RGB to BGR for display with OpenCV
        display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera Calibration', display_frame_bgr)

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
    cv2.destroyAllWindows()

    if images_captured == 0:
        print("No calibration images captured. Exiting.")
        return

    # Perform camera calibration
    print("\nCalculating calibration parameters...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    if ret:
        print(f"Calibration successful with RMS error: {ret}")

        # Calculate optimal camera matrix once
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

        # Save calibration parameters
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'optimal_camera_matrix': newcameramtx,
            'roi': roi,
            'image_shape': img_shape
        }

        with open(calibration_file, 'wb') as f:
            pickle.dump(calibration_data, f)

        print(f"Calibration parameters saved to {calibration_file}")

        # Test the calibration on a live feed
        test_calibration(mtx, dist, newcameramtx, roi)
    else:
        print("Calibration failed.")


def test_calibration(mtx, dist, newcameramtx, roi):
    """Test the calibration on a live camera feed"""
    print("\n==== TESTING CALIBRATION ====")
    print("Showing original and undistorted view side by side.")
    print("Press 'q' to quit, 's' to save a snapshot.")

    # Initialize Picamera2 again for testing
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        transform=Transform(hflip=True, vflip=True)  # Match the same transform used during calibration
    )
    picam2.configure(preview_config)
    picam2.start()

    # Allow camera to warm up
    time.sleep(1)

    # Extract ROI parameters
    x, y, w, h = roi

    snapshot_count = 0

    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Undistort the image
        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # Crop the undistorted image (optional)
        if all(val > 0 for val in [x, y, w, h]):
            undistorted = undistorted[y:y + h, x:x + w]
            # Resize undistorted to match original frame size for side-by-side display
            undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))

        # Add labels
        cv2.putText(frame, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(undistorted, "Undistorted", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert from RGB to BGR for display with OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        undistorted_bgr = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)

        # Display original and undistorted frames side by side
        combined = np.hstack((frame_bgr, undistorted_bgr))
        cv2.imshow('Calibration Test: Original | Undistorted', combined)

        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save a snapshot of the comparison
            snapshot_count += 1
            filename = f"calibration_test_{snapshot_count}.jpg"
            cv2.imwrite(filename, combined)
            print(f"Snapshot saved as {filename}")

    picam2.stop()
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
            # Load existing calibration data
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)

            mtx = calibration_data['camera_matrix']
            dist = calibration_data['dist_coeffs']

            # Check if we have the optimal camera matrix already
            if 'optimal_camera_matrix' in calibration_data and 'roi' in calibration_data:
                newcameramtx = calibration_data['optimal_camera_matrix']
                roi = calibration_data['roi']
            else:
                # Calculate it if not found
                img_shape = calibration_data['image_shape']
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

            # Test with existing calibration
            test_calibration(mtx, dist, newcameramtx, roi)
            return

    # Perform new calibration
    camera_calibration()
    print("\nCalibration process completed.")


if __name__ == "__main__":
    main()