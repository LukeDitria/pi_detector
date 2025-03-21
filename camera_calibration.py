import numpy as np
import cv2
import os
import pickle
import time


def camera_calibration():
    # Parameters
    board_size = (9, 6)  # Number of inner corners in the chessboard pattern
    square_size = 1.0  # Size in arbitrary units
    num_images_needed = 15  # Number of calibration images to capture
    calibration_file = "camera_calibration.pkl"

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get camera resolution
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame from camera.")
        cap.release()
        return

    img_shape = frame.shape[:2][::-1]  # (width, height)
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
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        # Create a copy of the frame for drawing
        display_frame = frame.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            if cv2.waitKey(1) & 0xFF == ord('c'):
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

    # Release camera before calibration
    cap.release()
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

        # Save calibration parameters
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'image_shape': img_shape
        }

        with open(calibration_file, 'wb') as f:
            pickle.dump(calibration_data, f)

        print(f"Calibration parameters saved to {calibration_file}")

        # Test the calibration on a live feed
        test_calibration(mtx, dist)
    else:
        print("Calibration failed.")


def test_calibration(mtx, dist):
    """Test the calibration on a live camera feed"""
    print("\n==== TESTING CALIBRATION ====")
    print("Showing original and undistorted view side by side.")
    print("Press 'q' to quit, 's' to save a snapshot.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera for testing.")
        return

    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        # Get optimal new camera matrix
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort the image
        undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # Crop the undistorted image (optional)
        x, y, w, h = roi
        if all(val > 0 for val in [x, y, w, h]):
            undistorted = undistorted[y:y + h, x:x + w]
            # Resize undistorted to match original frame size for side-by-side display
            undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))

        # Add labels
        cv2.putText(frame, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(undistorted, "Undistorted", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display original and undistorted frames side by side
        combined = np.hstack((frame, undistorted))
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the console
    print("===== CAMERA CALIBRATION TOOL =====")
    camera_calibration()
    print("\nCalibration process completed.")