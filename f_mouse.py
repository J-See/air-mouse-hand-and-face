import mediapipe as mp
import cv2
import numpy as np
from modules import face1 as fc
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
frameR = 150  # Frame Reduction
smoothening = 5
buttonPress = False

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
fc.faceDetector()
pTime = 0
cTime = 0

while cap.isOpened():
    success, img = cap.read()
    utils.textBlurBackground(
        img,
        "Face",
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (310, 50),
        2,
        utils.YELLOW,
        (71, 71),
        13,
        13,
    )
    img_height, img_width = img.shape[:2]
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fc.face_mesh.process(image=imgRGB)
    if results.multi_face_landmarks:
        mesh_coords = fc.landmarksDetection(img, results, False)
        distance = fc.depth(mesh_coords)
        if mesh_coords:
            ratio, leRatio, reRatio = fc.blinkRatio(
                mesh_coords, fc.RIGHT_EYE, fc.LEFT_EYE
            )
            # print(mesh_coords[8])
            x1, y1 = mesh_coords[8]
            img = utils.textWithBackground(
                img,
                f"Ratio: B={ratio:.2f}, L={leRatio:.2f}, R={reRatio:.2f}",
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 150),
                textThickness=2,
                bgColor=utils.PINK,
                textColor=utils.BLACK,
                bgOpacity=0.7,
                pad_x=6,
                pad_y=6,
            )

            img = utils.textWithBackground(
                img,
                f"Distance: {distance}",
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 185),
                textThickness=2,
                bgColor=utils.PINK,
                textColor=utils.BLACK,
                bgOpacity=0.7,
                pad_x=6,
                pad_y=6,
            )
            # print(x1, y1)
            # print(ratio, leRatio, reRatio)

            ## move cursor
            if mesh_coords[8] and distance < 65:
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
                cv2.rectangle(
                    img,
                    (frameR, frameR),
                    (wCam - frameR, hCam - frameR),
                    (255, 0, 255),
                    2,
                )
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                # convert coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Smoothen value
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

            ## left click
            if ((leRatio > 3.9) and not (ratio > 4.5)) and distance < 75:
                img = utils.fillPolyTrans(
                    img,
                    [np.array([mesh_coords[p] for p in fc.LEFT_EYE], dtype=np.int32)],
                    utils.GREEN,
                    0.5,
                )
                img = utils.textWithBackground(
                    img,
                    "Left eye blink => left click",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.YELLOW,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                print("left Click")
                pyautogui.leftClick()
                pyautogui.PAUSE = 0.2

            ## right click
            if (reRatio > 3.9 and not (ratio > 4.5)) and distance < 75:
                img = utils.fillPolyTrans(
                    img,
                    [np.array([mesh_coords[p] for p in fc.RIGHT_EYE], dtype=np.int32)],
                    utils.GREEN,
                    0.5,
                )
                img = utils.textWithBackground(
                    img,
                    "Right eye blink => ight click",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (textX, textY),
                    textThickness=2,
                    bgColor=utils.YELLOW,
                    textColor=utils.BLACK,
                    bgOpacity=0.7,
                    pad_x=6,
                    pad_y=6,
                )
                print(f"right click")
                pyautogui.rightClick()
                pyautogui.PAUSE = 0.2

            ##scroll mode
            if distance > 75:
                if (leRatio > 3.9) and not (ratio > 4.5):
                    img = utils.textWithBackground(
                        img,
                        "Scroll up",
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (textX, textY),
                        textThickness=2,
                        bgColor=utils.YELLOW,
                        textColor=utils.BLACK,
                        bgOpacity=0.7,
                        pad_x=6,
                        pad_y=6,
                    )
                    pyautogui.scroll(10)
                    print("up")
                if (reRatio > 3.9) and not (ratio > 4.5):
                    img = utils.textWithBackground(
                        img,
                        "Scroll down",
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (textX, textY),
                        textThickness=2,
                        bgColor=utils.YELLOW,
                        textColor=utils.BLACK,
                        bgOpacity=0.7,
                        pad_x=6,
                        pad_y=6,
                    )
                    pyautogui.scroll(-10)
                    print("Down")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
