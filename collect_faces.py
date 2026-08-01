import cv2
import os

# =================================
# ASK FOR PERSON NAME
# =================================

person_name = input("Enter person's name: ").strip()

if person_name == "":
    print("Name cannot be empty")
    exit()


# =================================
# CREATE PERSON FOLDER
# =================================

person_folder = os.path.join(
    "dataset",
    person_name
)

os.makedirs(
    person_folder,
    exist_ok=True
)


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
# VARIABLES
# =================================

count = 0
max_images = 30


# =================================
# CAMERA LOOP
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


    # Process faces
    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Save face
        if count < max_images:

            face = gray[
                y:y+h,
                x:x+w
            ]

            count += 1

            filename = os.path.join(
                person_folder,
                f"{count}.jpg"
            )

            cv2.imwrite(
                filename,
                face
            )

            print(
                f"Saved: {filename}"
            )


    # Display information
    cv2.putText(
        frame,
        f"Person: {person_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Images: {count}/{max_images}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Move face slowly",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show camera
    cv2.imshow(
        "Collect Face Dataset",
        frame
    )


    # Stop when complete
    if count >= max_images:
        print("Dataset collection completed!")
        break


    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
cap.release()
cv2.destroyAllWindows()