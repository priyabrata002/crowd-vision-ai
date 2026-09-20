import cv2
from ultralytics import YOLO

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Crowd Vision .AI - Anomaly Detection started.")
print("Press Q to exit.")


# Previous frame's zone counts
previous_counts = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 0
}


# How many additional people must appear
# before we flag a sudden increase.
ANOMALY_INCREASE = 3


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    height, width = frame.shape[:2]

    mid_x = width // 2
    mid_y = height // 2


    # Detect people
    results = model(
        frame,
        classes=[0],
        verbose=False
    )

    result = results[0]


    # Current zone counts
    zone_counts = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }


    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            if center_x < mid_x and center_y < mid_y:
                zone_counts["A"] += 1

            elif center_x >= mid_x and center_y < mid_y:
                zone_counts["B"] += 1

            elif center_x < mid_x and center_y >= mid_y:
                zone_counts["C"] += 1

            else:
                zone_counts["D"] += 1


    # Detect sudden increases
    anomalies = []

    for zone in zone_counts:

        increase = (
            zone_counts[zone]
            - previous_counts[zone]
        )

        if increase >= ANOMALY_INCREASE:

            anomalies.append(
                f"ZONE {zone}: SUDDEN CROWD INCREASE"
            )


    # Save current counts for next frame
    previous_counts = zone_counts.copy()


    # Draw detections
    annotated_frame = result.plot()


    # Draw zone lines
    cv2.line(
        annotated_frame,
        (mid_x, 0),
        (mid_x, height),
        (255, 255, 255),
        2
    )

    cv2.line(
        annotated_frame,
        (0, mid_y),
        (width, mid_y),
        (255, 255, 255),
        2
    )


    # Display zone counts
    cv2.putText(
        annotated_frame,
        f"A: {zone_counts['A']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"B: {zone_counts['B']}",
        (mid_x + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"C: {zone_counts['C']}",
        (20, mid_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"D: {zone_counts['D']}",
        (mid_x + 20, mid_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # Display anomaly warning
    if anomalies:

        cv2.putText(
            annotated_frame,
            "ANOMALY DETECTED!",
            (20, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        print(anomalies)


    else:

        cv2.putText(
            annotated_frame,
            "STATUS: NORMAL",
            (20, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Crowd Vision .AI - Anomaly Detection",
        annotated_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()