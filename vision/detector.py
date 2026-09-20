import cv2
from ultralytics import YOLO


# Load the small YOLO model.
# The "n" version is lightweight, so it is suitable for real-time webcam detection.
print("Loading YOLO...")
model = YOLO("yolo11n.pt")


# Open the computer's default webcam.
# Camera index 0 normally represents the primary camera.
camera = cv2.VideoCapture(0)


# Stop the program if Windows/OpenCV cannot access the camera.
if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()


print("Crowd Vision .AI - People Counter started.")
print("Press Q to exit.")


while True:

    # Capture one frame from the webcam.
    success, frame = camera.read()

    # If a frame could not be captured, stop safely.
    if not success:
        print("ERROR: Could not read frame.")
        break


    # Send the current frame to YOLO.
    # YOLO analyses the image and returns detected objects.
    results = model(frame, verbose=False)


    # Start the number of detected people at zero.
    people_count = 0


    # YOLO can return multiple result objects.
    # We process the first result because we supplied one frame.
    result = results[0]


    # Go through every detected object.
    for box in result.boxes:

        # YOLO assigns every detected object a class ID.
        class_id = int(box.cls[0])

        # COCO class ID 0 represents "person".
        # Therefore, we only count objects whose class is person.
        if class_id == 0:
            people_count += 1


    # Draw YOLO's bounding boxes and labels on the camera frame.
    annotated_frame = result.plot()


    # Display the current number of people on the screen.
    cv2.putText(
        annotated_frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # Show the processed camera feed.
    cv2.imshow("Crowd Vision .AI - People Counter", annotated_frame)


    # Press Q to close the camera window.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the webcam when the program finishes.
camera.release()

# Close all OpenCV windows.
cv2.destroyAllWindows()