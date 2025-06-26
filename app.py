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
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Set up the API key for Google AI
genai.configure(api_key="AIzaSyCI6Ng0XHh0Sxa9tShJi3ewvyRQ2Ot31mU")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam capture (use 0 for default webcam)
cap = None
while cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Try to use the default webcam (0)
    if not cap.isOpened():
        print("Waiting for webcam...")
        time.sleep(1)  # Wait for 1 second before retrying

cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Function to detect hand landmarks and return relevant information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

# Function to handle drawing based on hand gestures
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Gesture for drawing
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Gesture to clear canvas
        canvas = np.zeros_like(img)

    return current_pos, canvas

# Function to send the drawn canvas to AI model for math problem solving
def sendToAI(model, canvas, fingers):
    if fingers == [0, 0, 0, 0, 1]:  # Gesture to submit the drawn math problem
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text

# Initialize variables
prev_pos = None
canvas = None
output_text = ""

# Start capturing frames from the webcam and process them
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from webcam.")
        break
    img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Wait for the next frame
    cv2.waitKey(1)

cap.release()  # Release the webcam when done
cv2.destroyAllWindows()  # Close any OpenCV windows if opened
