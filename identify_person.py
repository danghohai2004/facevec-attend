from numpy.linalg import norm
from extract_embedding import extract_embeddings
import numpy as np

person_path = "doan4_myface"
database_embeddings = extract_embeddings(person_path)

def cosine_similarity(a, b):
    return np.dot(a, b)

def identify_person(embedding_frame, threshold):
    best_score = -1
    name_person_score = "Unknown"

    for person_name, embeddings_list in database_embeddings.items():
        scores = [cosine_similarity(embedding_frame, db_embedding) for db_embedding in embeddings_list]
        max_score = max(scores)

        if max_score > best_score:
            best_score = max_score
            name_person_score = person_name

    if best_score < threshold:
        name_person_score = "Unknown"

    return name_person_score