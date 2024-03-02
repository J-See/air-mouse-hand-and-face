# new branch with mediapipe hand model
from matplotlib import image
import mediapipe as mp
import cv2
import numpy as np
import numpy as np
import pyautogui
import autopy
import time
import streamlit as st
from modules import hand as hd
from modules import face as fc
from modules import utils

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
with open("./resources/style.css", 'r') as f:
    st.markdown(f"<style>{f.read()}</style>", True)

# Slidebar
st.sidebar.header("Hello Let's start `beta`")
app_mode = st.sidebar.selectbox("App Mode", ["About", "Hand", "Face"])


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


def main(
    frame,
    width,
    height,
    plocX,
    plocY,
    clocX,
    clocY,
    fps_input,
    wScr,
    hScr,
    frameR,
    smoothening,
    buttonPress,
    textX,
    textY,
):
    hands, img = hd.findHands(frame, draw=True, flipType=True)
    finger1 = []  # return
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"]
        handType1 = hand1["type"]
        if len(lmList1) != 0:
            x1, y1, _ = lmList1[8][:]  # index finger
            x2, y2, _ = lmList1[12][:]  # niddle finger
            finger1 = hd.fingersUp(hand1)

            ## move cursor
            if finger1 == [0, 1, 0, 0, 0]:
                utils.textBlurBackground(
                    img,
                    "Moving mode",
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (textX, textY),
                    2,
                    utils.GREEN,
                    (71, 71),
                    13,
                    13,
                )
                cv2.rectangle(
                    img,
                    (frameR, frameR),
                    (width - frameR, height - frameR),
                    (255, 0, 255),
                    2,
                )
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                x3 = np.interp(x1, (frameR, width - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, height - frameR), (0, hScr))

                # Smoothen value
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                autopy.mouse.move(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

            ## left click
            if finger1 == [0, 1, 1, 0, 0]:
                utils.textBlurBackground(
                    img,
                    "Left click mode",
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (textX, textY),
                    2,
                    utils.GREEN,
                    (71, 71),
                    13,
                    13,
                )
                length, info, img = hd.findDistance(
                    lmList1[8][0:2], lmList1[12][0:2], img
                )
                if length < 40:
                    cv2.circle(img, (info[-2], info[-1]), 10, (0, 255, 0), cv2.FILLED)
                    print("left Click")
                    pyautogui.leftClick()

            ## right click
            if finger1 == [1, 1, 0, 0, 0]:
                utils.textBlurBackground(
                    img,
                    "Right click mode",
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (textX, textY),
                    2,
                    utils.GREEN,
                    (71, 71),
                    13,
                    13,
                )
                length, info, img = hd.findDistance(
                    lmList1[4][0:2], lmList1[8][0:2], img
                )
                # print(length)
                if length < 30:
                    cv2.circle(img, (info[-2], info[-1]), 10, (0, 255, 0), cv2.FILLED)
                    print(f"right click")
                    pyautogui.rightClick()

            ## scroll mode
            if finger1 == [1, 1, 1, 1, 1]:
                utils.textBlurBackground(
                    img,
                    "Scroll mode",
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (textX, textY),
                    2,
                    utils.GREEN,
                    (71, 71),
                    13,
                    13,
                )
                cv2.line(
                    img=img,
                    pt1=(int(width / 4), int(height / 2) + 50),
                    pt2=(width - int(width / 4), int(height / 2) + 50),
                    color=(0, 255, 0),
                    thickness=3,
                )
                cv2.circle(
                    img=img,
                    center=(int(width / 2), int(height / 2) + 50),
                    radius=10,
                    color=(0, 0, 255),
                    thickness=cv2.FILLED,
                )

                lengthP, infoP = distance(lmList1[0][0:2], lmList1[9][0:2])
                center_of_wCam = int(width / 2), int(height / 2) + 50
                center_of_hand = infoP[-2], infoP[-1]
                scrollValue, info, img = distance(center_of_wCam, center_of_hand, img)

                if scrollValue > 68:
                    print("UP")
                    pyautogui.scroll(scrollValue)

                if scrollValue < -40:
                    print("DOWN")
                    pyautogui.scroll(scrollValue)

            ## left arrow click
            if finger1 == [0, 0, 0, 0, 0]:
                buttonPress = True
            if finger1 == [1, 0, 0, 0, 0]:
                utils.textBlurBackground(
                    img,
                    "Left arrow mode",
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (textX, textY),
                    2,
                    utils.GREEN,
                    (71, 71),
                    13,
                    13,
                )
                if buttonPress == True:
                    autopy.key.tap(autopy.key.Code.LEFT_ARROW)
                    print("LEFT KEYPRESS")
                    buttonPress = False

            ## right arrow click
            if finger1 == [0, 0, 0, 0, 0]:
                buttonPress = True
            if finger1 == [0, 0, 0, 0, 1]:
                utils.textBlurBackground(
                    img,
                    "Right arrow mode",
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (textX, textY),
                    2,
                    utils.GREEN,
                    (71, 71),
                    13,
                    13,
                )
                if buttonPress == True:
                    autopy.key.tap(autopy.key.Code.RIGHT_ARROW)
                    print("Right KEYPRESS")
                    buttonPress = False
    return (
        finger1,
        img,
        width,
        height,
        plocX,
        plocY,
        clocX,
        clocY,
        fps_input,
        wScr,
        hScr,
        frameR,
        smoothening,
        buttonPress,
        textX,
        textY,
        hands,
    )


## About
if app_mode == "About":
    st.sidebar.markdown(
        """
    ---
    Created with ❤️ by [J-See](https://github.com/J-See).
    """
    )
    with open("./resources/about_content.md", "r", encoding="utf-8") as f:
        about = f.read()
    st.markdown(f"{about}", unsafe_allow_html=True)
    ### instruction
    st.markdown("## Instruction", True)
    st.markdown("### Hand")
    colm_hand = st.columns(3)
    with colm_hand[0]:
        st.image("./resources/hand/move.jpeg")
        st.image("./resources/hand/left_click.png")
    with colm_hand[1]:
        st.image("./resources/hand/right_click.png")
        st.image("./resources/hand/left_arrow.png")
    with colm_hand[2]:
        st.image("./resources/hand/scroll.png")
        st.image("./resources/hand/right_arrow.png")
    st.markdown("### Face")
    colm_face = st.columns(3)
    with colm_face[0]:
        st.image(
            "./resources/face/face_M.png",
            "Distance between clent face and webcam should be less than 65cm, face should within rectangle",
        )
        st.image(
            "./resources/face/face_SU.png",
            "Distance between clent face and webcam must be greater than 75cm with left eye close",
        )
    with colm_face[1]:
        st.image(
            "./resources/face/face_LC.png",
            "Distance between clent face and webcam should be less than 75cm with left eye close",
        )
        st.image(
            "./resources/face/face_SD.png",
            "Distance between clent face and webcam must be greater than 75cm with right eye close",
        )
    with colm_face[2]:
        st.image(
            "./resources/face/face_RC.png",
            "Distance between clent face and webcam should be less than 75cm with right eye close",
        )


## Hand
if app_mode == "Hand":
    st.title("Control with hand 👌")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Drawing")
    textX = st.sidebar.slider("X:", 0, 640, 277)
    textY = st.sidebar.slider("Y:", 0, 480, 432)
    st.sidebar.markdown(
        """
    ---
    Created with ❤️ by [J-See](https://github.com/J-See).
    """
    )
    stframe = st.empty()
    video = cv2.VideoCapture(0)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))
    wScr, hScr = autopy.screen.size()
    frameR = 100  # Frame Reduction
    smoothening = 5
    buttonPress = False

    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    fps = 0
    i = 0
    prevTime = 0
    st.markdown("# Metrics")
    kpil, kpil2, kpil3 = st.columns(3)
    with kpil:
        st.subheader("Frame Rate")
        kpil_text = st.markdown("0")

    with kpil2:
        st.subheader("Hands")
        kpil2_text = st.markdown("0")

    with kpil3:
        st.subheader("Fingers")
        kpil3_text = st.markdown("0")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    hd.HandDetector()
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            st.write("the video capture is ended")
            break
        (
            finger1,
            img,
            width,
            height,
            plocX,
            plocY,
            clocX,
            clocY,
            fps_input,
            wScr,
            hScr,
            frameR,
            smoothening,
            buttonPress,
            textX,
            textY,
            hands,
        ) = main(
            frame,
            width,
            height,
            plocX,
            plocY,
            clocX,
            clocY,
            fps_input,
            wScr,
            hScr,
            frameR,
            smoothening,
            buttonPress,
            textX,
            textY,
        )

        # FPS Counter
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        # Dashboard
        kpil_text.write(f"{int(fps)}")
        kpil2_text.write(f"{len(hands)}")
        kpil3_text.write(f"{finger1}")
        stframe.image(frame, channels="BGR")


## Face
if app_mode == "Face":
    st.title("Control with face 😎")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Drawing")
    textX = st.sidebar.slider("X:", 0, 640, 277)
    textY = st.sidebar.slider("Y:", 0, 480, 432)
    st.sidebar.markdown(
        """
    ---
    Created with ❤️ by [J-See](https://github.com/J-See).
    """
    )
    stframe = st.empty()
    video = cv2.VideoCapture(0)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))
    wScr, hScr = autopy.screen.size()
    frameR = 100  # Frame Reduction
    smoothening = 5
    buttonPress = False

    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    fps = 0
    i = 0
    prevTime = 0
    st.markdown("# Metrics")
    kpil, kpil2, kpil3 = st.columns(3)
    with kpil:
        st.subheader("Frame Rate")
        kpil_text = st.markdown("0")

    with kpil2:
        st.subheader("Distance")
        kpil2_text = st.markdown("0")

    with kpil3:
        st.subheader("Ratio")
        kpil3_text = st.markdown("0")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    fc.FaceMeshDetector(
        staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5
    )
    while video.isOpened():
        success, img = video.read()
        img, faces, results = fc.findFaceMesh(img,flip=True, draw=False)
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
                    (width - frameR, height - frameR),
                    (255, 0, 255),
                    2,
                )
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                # convert coordinates
                x3 = np.interp(x1, (frameR, width - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, height - frameR), (0, hScr))

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

        # FPS Counter
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        # Dashboard
        kpil_text.write(f"{int(fps)}")
        # kpil2_text.write(f"{d:.2f}")
        # kpil3_text.write(f"{ratio:.2f}")
        stframe.image(img, channels="BGR")
