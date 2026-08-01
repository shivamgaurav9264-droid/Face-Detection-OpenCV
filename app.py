import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Face Detection with OpenCV",
    page_icon="👤",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("👤 Face Detection with OpenCV")

st.write(
    "Upload an image to detect faces, eyes, and smiles."
)


# ==========================================
# HAAR CASCADE FILES
# ==========================================

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "opencv/opencv/master/data/haarcascades/"
)


def download_cascade(filename):

    if not os.path.exists(filename):

        url = BASE_URL + filename

        urllib.request.urlretrieve(
            url,
            filename
        )

    return filename


# ==========================================
# DOWNLOAD CASCADES
# ==========================================

face_file = download_cascade(
    "haarcascade_frontalface_default.xml"
)

eye_file = download_cascade(
    "haarcascade_eye.xml"
)

smile_file = download_cascade(
    "haarcascade_smile.xml"
)


# ==========================================
# LOAD CASCADES
# ==========================================

face_cascade = cv2.CascadeClassifier(
    face_file
)

eye_cascade = cv2.CascadeClassifier(
    eye_file
)

smile_cascade = cv2.CascadeClassifier(
    smile_file
)


# ==========================================
# CHECK CASCADES
# ==========================================

if face_cascade.empty():

    st.error("Face detector could not be loaded.")
    st.stop()


if eye_cascade.empty():

    st.error("Eye detector could not be loaded.")
    st.stop()


if smile_cascade.empty():

    st.error("Smile detector could not be loaded.")
    st.stop()


# ==========================================
# UPLOAD IMAGE
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    # Read image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Convert PIL → NumPy
    frame = np.array(image)


    # RGB → BGR
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )


    # BGR → Grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================
    # FACE DETECTION
    # ======================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    # ======================================
    # PROCESS EACH FACE
    # ======================================

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
        roi_gray = gray[
            y:y+h,
            x:x+w
        ]

        roi_color = frame[
            y:y+h,
            x:x+w
        ]


        # ==================================
        # EYE DETECTION
        # ==================================

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


        # ==================================
        # SMILE DETECTION
        # ==================================

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


    # ======================================
    # FACE COUNT
    # ======================================

    face_count = len(faces)


    if face_count > 0:

        st.success(
            f"✅ {face_count} face(s) detected!"
        )

    else:

        st.warning(
            "No face detected."
        )


    # ======================================
    # CONVERT BACK TO RGB
    # ======================================

    result = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.image(
        result,
        caption="Detection Result",
        use_container_width=True
    )