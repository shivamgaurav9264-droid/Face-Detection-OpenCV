import cv2

# ==============================
# FACE DETECTOR
# ==============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==============================
# EYE DETECTOR
# ==============================

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# ==============================
# SMILE DETECTOR
# ==============================

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_smile.xml"
)


# Check detectors

if face_cascade.empty():
    print("Face detector not loaded")
    exit()

if eye_cascade.empty():
    print("Eye detector not loaded")
    exit()

if smile_cascade.empty():
    print("Smile detector not loaded")
    exit()


# ==============================
# OPEN CAMERA
# ==============================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ==============================
    # FACE DETECTION
    # ==============================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    face_count = len(faces)


    # ==============================
    # PROCESS FACES
    # ==============================

    for (x, y, w, h) in faces:

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==============================
        # FACE REGION
        # ==============================

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        # ==============================
        # EYE DETECTION
        # ==============================

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )


        # ==============================
        # SMILE DETECTION
        # ==============================

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:

            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 0, 255),
                2
            )

            cv2.putText(
                roi_color,
                "Smile",
                (sx, sy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


    # ==============================
    # FACE COUNT
    # ==============================

    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ==============================
    # SHOW WINDOW
    # ==============================

    cv2.imshow(
        "Face, Eye and Smile Detection",
        frame
    )


    # ==============================
    # EXIT
    # ==============================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# ==============================
# RELEASE CAMERA
# ==============================

cap.release()
cv2.destroyAllWindows()