import mediapipe as mp
import cv2
import numpy as np
from modules import hand as hd
from modules import utils
import numpy as np
import pyautogui
import autopy
import time

## scrolling distance
def distance(p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = -((x2 - x1) + (y2 - y1))
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

## Variable And Constant
textX = 117
textY = 462
wCam, hCam = 640, 480
# wScr, hScr = pyautogui.size()
wScr, hScr = autopy.screen.size()
frameR = 100  # Frame Reduction
smoothening = 5
buttonPress = False

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
hd.HandDetector()
pTime = 0
cTime = 0
while cap.isOpened():
    success, img = cap.read()
    utils.textBlurBackground(
        img,
        "Hand",
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (310, 50),
        2,
        utils.YELLOW,
        (71, 71),
        13,
        13,
    )
    hands, img = hd.findHands(img) # find hands and give hand landmarks of it
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1['center']
        handType1 = hand1["type"]
        if len(lmList1) != 0:
            x1, y1, _ = lmList1[8][:]  # index finger
            x2, y2, _ = lmList1[12][:]  # niddle finger 
            finger1 = hd.fingersUp(hand1)
            
            ## move cursor
            if finger1 == [0, 1, 0, 0, 0]:
                img = utils.textWithBackground(
                    img,
                    "Moving mode",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.GREEN,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Smoothen value
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY
                
            ## left click
            if finger1 == [0, 1, 1, 0, 0]:
                img = utils.textWithBackground(
                    img,
                    "Left click mode",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.GREEN,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                length, info, img = hd.findDistance(lmList1[8][0:2], lmList1[12][0:2], img)
                if length < 40:
                    cv2.circle(img, (info[-2], info[-1]), 10, (0, 255, 0), cv2.FILLED)
                    print("left Click")
                    pyautogui.leftClick()
                    
            ## right click
            if finger1 == [1, 1, 0, 0, 0]:
                img = utils.textWithBackground(
                    img,
                    "Right click mode",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.GREEN,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                length, info, img = hd.findDistance(lmList1[4][0:2], lmList1[8][0:2], img)
                # print(length)
                if length < 30:
                    cv2.circle(img, (info[-2], info[-1]), 10, (0, 255, 0), cv2.FILLED)
                    print(f"right click")
                    pyautogui.rightClick()
                
            ## scroll mode
            if finger1 == [1, 1, 1, 1, 1]:
                img = utils.textWithBackground(
                    img,
                    "scrollimg mpde",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.GREEN,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                cv2.line(img=img, pt1=(int(wCam/4), int(hCam/2) + 50), pt2=(wCam - int(wCam / 4), int(hCam / 2) + 50), color=(0, 255, 0), thickness=3)
                cv2.circle(img=img, center=(int(wCam / 2), int(hCam / 2) + 50), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
                
                lengthP, infoP= distance(lmList1[0][0:2], lmList1[9][0:2])
                center_of_wCam = int(wCam / 2), int(hCam / 2) + 50
                center_of_hand = infoP[-2], infoP[-1]
                scrollValue, info, img = distance(center_of_wCam, center_of_hand, img)
                
                if scrollValue > 68:                    
                    print('UP')
                    pyautogui.scroll(scrollValue)
                    
                if scrollValue < -40:
                    print('DOWN')
                    pyautogui.scroll(scrollValue)
            
            ## left arrow click
            if finger1 == [0, 0, 0, 0, 0]:
                buttonPress = True
            if finger1 == [1, 0, 0, 0, 0]:
                img = utils.textWithBackground(
                    img,
                    "<-- mode",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.GREEN,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                if buttonPress == True:
                    autopy.key.tap(autopy.key.Code.LEFT_ARROW)
                    print("LEFT KEYPRESS")
                    buttonPress = False
            
            ## right arrow click
            if finger1 == [0, 0, 0, 0, 0]:
                buttonPress = True
            if finger1 == [0, 0, 0, 0, 1]:
                img = utils.textWithBackground(
                    img,
                    "--> mode",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.GREEN,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                if buttonPress == True:
                    autopy.key.tap(autopy.key.Code.RIGHT_ARROW)
                    print("Right KEYPRESS")
                    buttonPress = False
                

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
