import cv2

# =================================
# FACE DETECTOR
# =================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Face detector could not be loaded")
    exit()


# =================================
# LOAD TRAINED MODEL
# =================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    "face_model.yml"
)


# =================================
# LOAD NAMES
# =================================

names = {}

with open(
    "names.txt",
    "r"
) as file:

    for line in file:

        label, name = line.strip().split(":")

        names[int(label)] = name


# =================================
# OPEN CAMERA
# =================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()


# =================================
# MAIN LOOP
# =================================

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


    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )


    # Process each face
    for (x, y, w, h) in faces:

        face = gray[
            y:y+h,
            x:x+w
        ]


        # Predict
        label, confidence = recognizer.predict(
            face
        )


        # =================================
        # RECOGNIZE PERSON
        # =================================

        if confidence < 70:

            name = names.get(
                label,
                "Unknown"
            )

        else:

            name = "Unknown"


        # =================================
        # DRAW BOX
        # =================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # =================================
        # DISPLAY NAME
        # =================================

        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        # Display confidence
        cv2.putText(
            frame,
            f"Distance: {confidence:.2f}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # =================================
    # TITLE
    # =================================

    cv2.putText(
        frame,
        "Multi-Person Face Recognition",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # =================================
    # SHOW
    # =================================

    cv2.imshow(
        "Face Recognition",
        frame
    )


    # =================================
    # EXIT
    # =================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# =================================
# RELEASE
# =================================

cap.release()
cv2.destroyAllWindows()