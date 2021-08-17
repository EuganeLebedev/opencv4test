import os

import cv2
import face_recognition
import numpy as np

path = "6_img/known_faces"
images = []
class_names = []
photo_list = os.listdir(path)
print(photo_list)

for cls in photo_list:
    cur_img = cv2.imread(f'{path}/{cls}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cls)[0])

def find_encodings(images):
    encode_list=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

encode_list_known = find_encodings(images)

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    # imgS = cv2.resize(img, (0,0), None, 0.25, 0,25)
    imgS = img
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    face_cur_frame = face_recognition.face_locations(imgS)
    encode_cur_frame = face_recognition.face_encodings(imgS, face_cur_frame)

    for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        print(face_dis)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = class_names[match_index]

            print(f"MATCH! {name}")
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow("WebCam", img)
    cv2.waitKey(1)


def face_rec():
    pass

def main():
    face_rec()



if __name__ == '__main__':
    main()