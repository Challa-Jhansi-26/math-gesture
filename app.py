import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import time

st.set_page_config(layout="wide")

# Streamlit columns
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Set up the API key for Google AI
genai.configure(api_key="AIzaSyCI6Ng0XHh0Sxa9tShJi3ewvyRQ2Ot31mU")  # Replace with your actual key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Drawing function
def draw(fingers, lmList, prev_pos, canvas, img):
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Draw
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Clear
        canvas = np.zeros_like(img)
    return current_pos, canvas

# Send canvas to Gemini
def sendToAI(model, canvas):
    pil_image = Image.fromarray(canvas)
    response = model.generate_content(["Solve this math problem", pil_image])
    return response.text

# Run if checkbox is selected
if run:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    canvas = None
    prev_pos = None
    output_text = ""

    while run:
        success, img = cap.read()
        if not success:
            st.error("Webcam not working.")
            break

        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        hands, img = detector.findHands(img, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)

            prev_pos, canvas = draw(fingers, lmList, prev_pos, canvas, img)

            if fingers == [0, 0, 0, 0, 1]:  # Submit gesture
                output_text = sendToAI(model, canvas)

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

        if output_text:
            output_text_area.text(output_text)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
