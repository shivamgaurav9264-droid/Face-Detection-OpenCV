import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Face Detection with OpenCV")
st.write("Upload an image to detect faces, eyes, and smiles.")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Load eye detector
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# Load smile detector
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_smile.xml"
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)

    # Convert PIL image to OpenCV format
    frame = np.array(image)

    # Convert RGB to BGR
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

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
        minSize=(30, 30)
    )

    # Process each face
    for (x, y, w, h) in faces:

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

        # Detect smile
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20
        )

        for (sx, sy, sw, sh) in smiles:

            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 0, 255),
                2
            )

    # Face count
    face_count = len(faces)

    st.success(
        f"Faces detected: {face_count}"
    )

    # Convert BGR back to RGB
    result = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Display result
    st.image(
        result,
        caption="Detection Result",
        use_container_width=True
    )