import cv2

frontal_face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
eyes_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('Images/10.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = frontal_face_cascade_db.detectMultiScale(img_grey, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    eyes = eyes_cascade_db.detectMultiScale(img_grey[y:y + h, x:x + w], 1.1, 10)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 1)

cv2.imshow('result', img)
cv2.waitKey()
