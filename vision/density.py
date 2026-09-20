import cv2
from ultralytics import YOLO

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Crowd Vision .AI - Density Analysis started.")
print("Press Q to exit.")


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    results = model(
        frame,
        classes=[0],
        verbose=False
    )

    result = results[0]

    people_count = 0

    if result.boxes is not None:
        people_count = len(result.boxes)

    # Initial prototype density classification
    if people_count <= 2:
        density = "LOW"
    elif people_count <= 5:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # Draw detections
    annotated_frame = result.plot()

    # Display people count
    cv2.putText(
        annotated_frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display density
    cv2.putText(
        annotated_frame,
        f"Density: {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Crowd Vision .AI - Density Analysis",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()