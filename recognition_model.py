import os
import pickle
import face_recognition
from cv2 import cv2


# Функция, которая обрабатывает лица в датасете и тем самым обучает модель
def train_model_by_img(name):

    known_encodings = []
    images = os.listdir("dataset")
    for (i, image) in enumerate(images):
        face_img = face_recognition.load_image_file(f"dataset/{image}")
        face_enc = face_recognition.face_encodings(face_img)[0]

        if len(known_encodings) == 0:
            known_encodings.append(face_enc)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])

                if result [0]:
                    known_encodings.append(face_enc)
                    break
                else:
                    break
    data = {
        "name": name,
        "encodings": known_encodings
    }

    with open(f"{name}_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))
    return f"[INFO] Файл {name}_encodings.pickle успешно создан"


def main():
    train_model_by_img("leo")


if __name__ == '__main__':
    main()
