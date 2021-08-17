import cv2

frontal_face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
#eyes_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # bodies = body_cascade_db.detectMultiScale(frame_grey, 1.1, 5)
    faces = frontal_face_cascade_db.detectMultiScale(frame_grey, 1.1, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
