import cv2
import mediapipe as mp
import time
import math

def HandDetector(minDtection = 0.5, minTracking = 0.5):
    global mpHolistic, holistic, mpDraw, tipIds, fingers, lmList
    mpHolistic = mp.solutions.holistic
    holistic = mpHolistic.Holistic(
        min_detection_confidence=minDtection, min_tracking_confidence=minTracking
    )
    mpDraw = mp.solutions.drawing_utils
    tipIds = [4, 8, 12, 16, 20]
    fingers = []
    lmList = []

def findHands(img, draw=True, flipType=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    global results, myHand
    results = holistic.process(imgRGB)
    allHands = []
    h, w, c = img.shape

    ## left hand
    if results.left_hand_landmarks:
        myHand = {}
        ## lmList
        mylmList = []
        xList = []
        yList = []
        for id, lm in enumerate(results.left_hand_landmarks.landmark):
            px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
            mylmList.append([px, py, pz])
            xList.append(px)
            yList.append(py)
        
        ## bbox
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH
        cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
        myHand["lmList"] = mylmList
        myHand["bbox"] = bbox
        myHand["center"] = (cx, cy)

        if flipType:
            if results.left_hand_landmarks:
                myHand["type"] = "Left"
            else:
                myHand["type"] = "Right"
        else:
            myHand["type"] = "Right"
        allHands.append(myHand)

        ## draw
        if draw:
            mpDraw.draw_landmarks(img, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                            (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                            (255, 0, 255), 2)
            cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
    
    ## right hand
    if results.right_hand_landmarks:
        myHand = {}
        ## lmList
        mylmList = []
        xList = []
        yList = []
        for id, lm in enumerate(results.right_hand_landmarks.landmark):
            px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
            mylmList.append([px, py, pz])
            xList.append(px)
            yList.append(py)
        
        ## bbox
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH
        cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
        myHand["lmList"] = mylmList
        myHand["bbox"] = bbox
        myHand["center"] = (cx, cy)

        if flipType:
            if results.right_hand_landmarks:
                myHand["type"] = "Right"
            else:
                myHand["type"] = "Left"
        else:
            myHand["type"] = "Left"
        allHands.append(myHand)

        ## draw
        if draw:
            mpDraw.draw_landmarks(img, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                            (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                            (255, 0, 255), 2)
            cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
            
    return allHands, img

def fingersUp(myHand):
    fingers = []
    myHandType = myHand["type"]
    myLmList = myHand["lmList"]

    if (results.left_hand_landmarks) or (results.right_hand_landmarks):
        # Thumb
        if myHandType == "Right":
            if myLmList[tipIds[0]][0] > myLmList[tipIds[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if myLmList[tipIds[0]][0] < myLmList[tipIds[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if myLmList[tipIds[id]][1] < myLmList[tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)


    return fingers

def findDistance(p1, p2, img=None, color=(255, 0, 255), scale=5):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
        cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
        cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
        cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

    return length, info, img

def main():
    cap = cv2.VideoCapture(0)
    HandDetector()
    pTime = 0
    cTime = 0
    while cap.isOpened():
        success, img = cap.read()
        hands, img = findHands(img, draw=True, flipType=True)
        print(hands)
        if hands:
            hand1 = hands[0]

            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            center1 = hand1['center']
            handType1 = hand1["type"]

            # Count the number of fingers up for the first hand
            fingers1 = fingersUp(hand1)       
            print(fingers1)
            # print(f'H1 = {fingers1.count(1)}', end=" ") # Print the count of fingers that are up
            
            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmarks points
                bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
                centerPoint2 = hand2["center"]  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type Left or Right

                fingers2 = fingersUp(hand2)
                print(fingers1, fingers2)
                # print(lmLis1t[8], lmList2[8])
                length, info, img = findDistance(lmList1[8][:2], lmList2[8][:2], img, color=(255, 0, 255), scale=10)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
