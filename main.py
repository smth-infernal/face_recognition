import face_recognition
import pickle
from cv2 import cv2


# Функция поиска нужного лица в видео
def detect_person_in_video():
    data = pickle.loads(open("leo_encodings.pickle", "rb").read())
    video = cv2.VideoCapture("video_tom_hardy.mp4")

    while True:
        ret, image = video.read()

        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encodings"], face_encoding)
            match = None

            if True in result:
                match = data["name"]
                print(f'Совпадение найдено! {match}')
            else:
                print('Объект не найден')

            left_top = (face_location[3], face_location[0])
            right_bottom = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color, 2)

            left_bottom = (face_location[3], face_location[2])
            right_bottom = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
            cv2.putText(
                image,
                match,
                (face_location[3] + 10, face_location[2] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4
            )

        cv2.imshow("Запущено распознавание лиц в видео", image)

        k = cv2.waitKey(1)
        if k == ord("q"):
            print('Нажата клавиша Q, закрытие программы')
            break


def main():
    detect_person_in_video()


if __name__ == '__main__':
    main()
