import cv2
from ultralytics import YOLO

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Crowd Vision .AI - Zone Analysis started.")
print("Press Q to exit.")


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Divide the camera into 4 zones
    mid_x = width // 2
    mid_y = height // 2

    zones = {
        "A": (0, 0, mid_x, mid_y),
        "B": (mid_x, 0, width, mid_y),
        "C": (0, mid_y, mid_x, height),
        "D": (mid_x, mid_y, width, height)
    }

    # Detect people
    results = model(
        frame,
        classes=[0],
        verbose=False
    )

    result = results[0]

    # Start every zone at zero people
    zone_counts = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Find center of detected person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Determine which zone the person belongs to
            if center_x < mid_x and center_y < mid_y:
                zone_counts["A"] += 1

            elif center_x >= mid_x and center_y < mid_y:
                zone_counts["B"] += 1

            elif center_x < mid_x and center_y >= mid_y:
                zone_counts["C"] += 1

            else:
                zone_counts["D"] += 1

    # Draw YOLO boxes
    annotated_frame = result.plot()

    # Draw vertical dividing line
    cv2.line(
        annotated_frame,
        (mid_x, 0),
        (mid_x, height),
        (255, 255, 255),
        2
    )

    # Draw horizontal dividing line
    cv2.line(
        annotated_frame,
        (0, mid_y),
        (width, mid_y),
        (255, 255, 255),
        2
    )

    # Display zone information
    cv2.putText(
        annotated_frame,
        f"A: {zone_counts['A']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"B: {zone_counts['B']}",
        (mid_x + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"C: {zone_counts['C']}",
        (20, mid_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"D: {zone_counts['D']}",
        (mid_x + 20, mid_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Crowd Vision .AI - Zone Analysis",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()