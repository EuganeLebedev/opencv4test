import cv2


frontal_face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')


img = cv2.imread('Images/7.jpg')

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

frontal_face_dataset = frontal_face_cascade_db.detectMultiScale(img_grey, 1.1, 1)

for (x, y, w, h) in frontal_face_dataset:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 1)




cv2.imshow('result', img)
cv2.waitKey()


