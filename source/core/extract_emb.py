import os
import cv2
from utils.model_app import setup_face_app

# Extract facial features into dictionary
def extract_embeddings(original_img_path, name):
    app = setup_face_app()
    db_embeddings = {}

    for person_name in os.listdir(original_img_path):
        if person_name == name:
            person_name_path = os.path.join(original_img_path, person_name)

            embeddings = []

            for img_name in os.listdir(person_name_path):
                img_path = os.path.join(person_name_path, img_name)
                img = cv2.imread(img_path)

                faces = app.get(img)

                embeddings.append(faces[0].normed_embedding)

            db_embeddings[person_name] = embeddings

    return db_embeddings
