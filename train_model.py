import cv2
import os
import numpy as np

# =================================
# DATASET PATH
# =================================

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    print("Dataset folder does not exist!")
    exit()


# =================================
# FACE DETECTOR
# =================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =================================
# VARIABLES
# =================================

faces = []
labels = []

label_names = {}

current_label = 0


# =================================
# READ PERSON FOLDERS
# =================================

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue


    current_label += 1

    label_names[current_label] = person_name

    print(
        f"Loading: {person_name} "
        f"-> Label {current_label}"
    )


    # Read images
    for filename in os.listdir(person_folder):

        if not filename.lower().endswith(".jpg"):
            continue


        image_path = os.path.join(
            person_folder,
            filename
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is not None:

            faces.append(image)

            labels.append(current_label)


# =================================
# CHECK DATA
# =================================

if len(faces) == 0:
    print("No training images found!")
    exit()


print(
    f"Total training images: {len(faces)}"
)

print(
    f"Total people: {len(label_names)}"
)


# =================================
# CREATE RECOGNIZER
# =================================

recognizer = cv2.face.LBPHFaceRecognizer_create()


# =================================
# TRAIN
# =================================

recognizer.train(
    faces,
    np.array(labels)
)


# =================================
# SAVE MODEL
# =================================

recognizer.write(
    "face_model.yml"
)


# =================================
# SAVE LABEL NAMES
# =================================

with open(
    "names.txt",
    "w"
) as file:

    for label, name in label_names.items():

        file.write(
            f"{label}:{name}\n"
        )


print("Training completed!")
print("Model saved as face_model.yml")
print("Names saved as names.txt")