import cv2
from pixellib.instance import instance_segmentation



instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")


#capture = cv2.VideoCapture('rtsp://admin:admin@192.168.200.13:8554/1')
capture = cv2.VideoCapture('rtsp://admin:admin@192.168.200.14/1')

target_classes = instance_seg.select_target_classes(person=True)

while capture.isOpened():
    ret, frame = capture.read()
    res = instance_seg.segmentFrame(frame, show_bboxes=True, segment_target_classes=target_classes)
    image = res[1]




    cv2.imshow('frame', image)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
