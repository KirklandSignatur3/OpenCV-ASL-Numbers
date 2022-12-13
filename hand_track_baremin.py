import cv2
import mediapipe as mp
import time

### Create Video Object ### 
cap = cv2.VideoCapture(0) # the field determines what video capture device we use.


### Create Hands Object ###
# basially a formality for the package, doesnt mean much.
mpHands = mp.solutions.hands
#? why wont parameters show up????
hands = mpHands.Hands() 
    # bool static mode: either detects or tracks, true means that it always tracks. 
    ## this is slow, so we are keeping it false
# mediapipe method for helping draw the hand landmarks
mpDraw = mp.solutions.drawing_utils

### For FPS tracking ###
pTime = 0
cTime = 0


####### This while loop runs while the camera is active ##########
while True:
    success, img = cap.read() #RHS gives us our frame. s
        #success = boolean of if an image was successfully captured
        #img = the actual image
    
    ### create an image readable by the CV model ###
    # concerts image to rgb image, class only uses rgb images
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # gets the results from the model processing the image
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) 
    #   Prints None when there are no hands detected
    #   Otherwise prints the coordonates of the hand landmarks 
    # Draws the hand landmakrs in the video image
    if results.multi_hand_landmarks:
        # gets list of landmarks per hand (landmakr object), from the results
        for handLandMarks in results.multi_hand_landmarks:
            # gets the "id" and "lm" params from the landmark object
            for id, lm in enumerate(handLandMarks.landmark):
                # print(id,lm)
                img_h, img_w, img_c = img.shape
                cx, cy = int(lm.x * img_w), int(lm.y * img_h)
                #print (id, cx, cy)

                # example of finding one landmark id and doing something with it
                if id == 0:
                    cv2.circle(img, (cx,cy), 30, (255, 0, 255),  cv2.FILLED)


            # daraws the hand landmark in the original image 
            #   we write "img" because we are drawing on the original image
            mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)

    ### code for FPS tracking ###
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # puts displays the fps in the camera
    cv2.putText(img, str("fps:{}".format(int(fps))), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


