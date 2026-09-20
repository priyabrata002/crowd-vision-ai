import cv2
from ultralytics import YOLO


print("Loading YOLO...")
model = YOLO("yolo11n.pt")


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()


print("Crowd Vision .AI - Person Tracking started.")
print("Press Q to exit.")


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break


    # Run YOLO tracking.
    # persist=True tells YOLO to maintain the same tracking IDs
    # across consecutive video frames.
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )


    result = results[0]

    people_count = 0


    if result.boxes is not None:

        people_count = len(result.boxes)


    # Draw bounding boxes and tracking IDs.
    annotated_frame = result.plot()


    cv2.putText(
        annotated_frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Crowd Vision .AI - Person Tracking",
        annotated_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()