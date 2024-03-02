import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy
import pyautogui


def FaceMeshDetector(
    staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5
):
    staticMode = staticMode
    maxFaces = maxFaces
    minDetectionCon = minDetectionCon
    minTrackCon = minTrackCon
    global mpDraw, mpFaceMesh, faceMesh, drawSpec
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(
        static_image_mode=staticMode,
        max_num_faces=maxFaces,
        min_detection_confidence=minDetectionCon,
        min_tracking_confidence=minTrackCon,
    )
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


def findFaceMesh(img, flip=True, draw=True):
    if flip:
        img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    faces = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            if draw:
                mpDraw.draw_landmarks(
                    img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec
                )
            face = []
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                face.append([x, y])
            faces.append(face)
    return img, faces, results



def main():
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
    cap = cv2.VideoCapture(0)
    FaceMeshDetector(
        staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5
    )

    while True:
        success, img = cap.read()
        img, faces, results = findFaceMesh(img,flip=True, draw=False)

        if faces:
            for face in faces:
                leftEyeUpPoint = face[159]
                leftEyeDownPoint = face[23]
                # leftEyeVerticalDistance, info = detector.findDistance(leftEyeUpPoint, leftEyeDownPoint)

                # print(leftEyeVerticalDistance)
                # print(f"up_point: {leftEyeUpPoint}, down_point: {leftEyeDownPoint}")
                x1, y1 = face[8]
                # cv2.circle(img, (x1, y1), 5, (255, 24, 255), cv2.FILLED)
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
                # left point
                left = [face[145], face[159]]
                for i in left:
                    x, y = i
                    cv2.circle(img, (x, y), 3, (0, 255, 255))
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    l = [landmarks[145], landmarks[159]]
                    print(l[0].y - l[1].y)
                    if (l[0].y - l[1].y) < 0.004:
                        pyautogui.click()
                        pyautogui.sleep(1)

        cv2.imshow('frame', img)
        key = cv2.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
