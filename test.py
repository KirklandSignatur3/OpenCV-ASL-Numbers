import cv2
# from hand_tracking_module import handDetector
from hand_tracking_module import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

detector = HandDetector(maxHands=2)
vid_cap = cv2.VideoCapture(0)
classifier = Classifier("Model\keras_model.h5", "Model\labels.txt")
offset = 20
imgSize = 300
counter = 0



while True:
    success, img = vid_cap.read() #RHS gives us our frame. s
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox'] # gets the bounding box from the hand oject

            #creates another image thats the size of the hand boundary box
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] #start:end 
        imgCropShape = imgCrop.shape
 
        aspectRatio = h / w

        if aspectRatio > 1: ## when the width is too small
            k = imgSize / h #sidth value of stretched out to 300
            wCal = math.ceil(k * w) #calulated width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize)) #rsize the original cropped image
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2) #gap of empty space in the cropped
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print( index)
            
 
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, str(index), (x,y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), 2 )

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        


    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
