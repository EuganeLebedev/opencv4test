import cv2

frontal_face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
eyes_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


#capture = cv2.VideoCapture('rtsp://admin:admin@192.168.200.13:8554/1')
capture = cv2.VideoCapture('rtsp://admin:admin@192.168.200.14/1')

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
