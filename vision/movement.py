import cv2
from ultralytics import YOLO

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Crowd Vision .AI - Movement Tracking started.")
print("Press Q to exit.")

# Stores the previous center position of each person
previous_positions = {}


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy
        track_ids = result.boxes.id

        for box, track_id in zip(boxes, track_ids):

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # Calculate center of the person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            person_id = int(track_id)

            # Default movement status
            direction = "STATIONARY"

            # Check whether we have seen this person before
            if person_id in previous_positions:

                old_x, old_y = previous_positions[person_id]

                dx = center_x - old_x
                dy = center_y - old_y

                # Ignore tiny movements caused by detection noise
                threshold = 5

                if abs(dx) > threshold or abs(dy) > threshold:

                    if abs(dx) > abs(dy):

                        if dx > 0:
                            direction = "RIGHT"
                        else:
                            direction = "LEFT"

                    else:

                        if dy > 0:
                            direction = "DOWN"
                        else:
                            direction = "UP"

            # Save current position for the next frame
            previous_positions[person_id] = (center_x, center_y)

            # Draw the person's center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )

            # Display ID and direction
            label = f"ID {person_id}: {direction}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Crowd Vision .AI - Movement Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()