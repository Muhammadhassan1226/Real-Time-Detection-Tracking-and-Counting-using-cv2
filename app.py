import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import math
from sort import *
# cap  = cv2.VideoCapture(0)
cap  = cv2.VideoCapture("vid/cars.mp4")


# cap.set(3, 1280)
# cap.set(4, 720)

model = YOLO('yolov8n.pt')

coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


mask  = cv2.imread("images/mask.png")


# tracking

tracker  = Sort(max_age=20,min_hits=2, iou_threshold=0.3)


#line

limits = [280, 500, 1200, 500]



totalCount = []

while True:
    success, image = cap.read()
    imgRegion = cv2.bitwise_and(image,mask)


    imgGraphic  = cv2.imread("images/card_card.png",cv2.IMREAD_UNCHANGED)
    image = cvzone.overlayPNG(image,imgGraphic,(0,0))
    results  = model(imgRegion,stream=True)
    # results  = model(image,stream=True)
    names = model.names
    print(names)

    detections  =  np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # border boxing

            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 =  int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            # cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),3)

            w,h = x2-x1,y2-y1


            # confidence

            conf = math.ceil((box.conf[0]*100))/100
            # conf = box.conf[0]
            print(conf)

            # class

            cls = int(box.cls[0])


            currentCLass = coco_classes[cls]
            if currentCLass == "car" or currentCLass == "truck" or currentCLass == "motorbike"\
                or currentCLass == "bus"  or currentCLass == "bicycle" and conf > 0.3:
                # cvzone.putTextRect(image,f'{coco_classes[cls]} {conf}',(max(0,x1), max(35,y1)), scale=0.9, thickness=2,offset=3)
                # cvzone.cornerRect(image, (x1, y1, w, h), l=9)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections  = np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),8)

    for ets in resultsTracker:
        x1,y1,x2,y2,Id = ets
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(image, (x1, y1, w, h), l=9,rt=3,colorR=(255,0,255))
        cvzone.putTextRect(image, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=1.5, thickness=2,
                           offset=5)

        # center for obj

        cx,cy  = x1 + w//2, y1 + h//2
        cv2.circle(image,(cx,cy),5,(255,0,255),5)


        if limits[0]  < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20 :
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 8)


        # cvzone.putTextRect(image, f'{len(totalCount)}', (50,50))
        cv2.putText(image,str(len(totalCount)), (255,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        print(ets)

    cv2.imshow("Image", image)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
