import cv2
IPHONE_URL = 1 
def test_camera():
    print("Trying to connect to iPhone camera...")
    print(f"URL: {IPHONE_URL}")

    # Try to open the camera stream
    cap = cv2.VideoCapture(IPHONE_URL)

    # Check if connection worked
    if not cap.isOpened():
        print("ERROR: Could not connect to camera.")
        print("Make sure:")
        print("  1. Your iPhone and laptop are on the SAME WiFi")
        print("  2. The app is open and running on your iPhone")
        print("  3. The URL above is correct")
        return

    print("SUCCESS: Camera connected!")
    print("A window will open showing your camera feed.")
    print("Press Q to quit.")

    # Show the camera feed in a window
    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Lost connection to camera.")
            break

        # Show the frame in a window called "iPhone Camera Test"
        cv2.imshow("iPhone Camera Test", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Test finished.")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()