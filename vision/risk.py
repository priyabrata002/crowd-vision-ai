import cv2
from ultralytics import YOLO


# -----------------------------
# Configuration
# -----------------------------

LOW_LIMIT = 2
MEDIUM_LIMIT = 5


# -----------------------------
# Load YOLO
# -----------------------------

print("Loading YOLO...")
model = YOLO("yolo11n.pt")


# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()


print("Crowd Vision .AI - Risk Engine started.")
print("Press Q to exit.")


# -----------------------------
# Risk function
# -----------------------------

def get_risk(count):

    if count <= LOW_LIMIT:
        return "LOW"

    elif count <= MEDIUM_LIMIT:
        return "MEDIUM"

    else:
        return "HIGH"


# -----------------------------
# Main loop
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break


    height, width = frame.shape[:2]

    mid_x = width // 2
    mid_y = height // 2


    # -----------------------------
    # Detect people
    # -----------------------------

    results = model(
        frame,
        classes=[0],
        verbose=False
    )

    result = results[0]


    # -----------------------------
    # Zone counters
    # -----------------------------

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


    # -----------------------------
    # Calculate risk
    # -----------------------------

    zone_risk = {}

    for zone, count in zone_counts.items():

        zone_risk[zone] = get_risk(count)


    # -----------------------------
    # Draw detections
    # -----------------------------

    annotated_frame = result.plot()


    # -----------------------------
    # Draw zone lines
    # -----------------------------

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


    # -----------------------------
    # Display zone information
    # -----------------------------

    cv2.putText(
        annotated_frame,
        f"A: {zone_counts['A']} | {zone_risk['A']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"B: {zone_counts['B']} | {zone_risk['B']}",
        (mid_x + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"C: {zone_counts['C']} | {zone_risk['C']}",
        (20, mid_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"D: {zone_counts['D']} | {zone_risk['D']}",
        (mid_x + 20, mid_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # -----------------------------
    # Show result
    # -----------------------------

    cv2.imshow(
        "Crowd Vision .AI - Risk Engine",
        annotated_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()