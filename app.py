import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from cvzone.HandTrackingModule import HandDetector

# Setup Google Gemini API
genai.configure(api_key="AIzaSyCI6Ng0XHh0Sxa9tShJi3ewvyRQ2Ot31mU")
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit UI
st.set_page_config(layout="wide")
col1, col2 = st.columns([3, 2])

with col2:
    st.title("Answer")
    output_text_area = st.empty()

with col1:
    run = st.checkbox("Run", value=True)

# Global canvas and result
canvas_global = None
output_global = ""


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7)
        self.prev_pos = None
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    def draw(self, fingers, lmList):
        current_pos = None

        if fingers == [0, 1, 0, 0, 0]:  # Index finger up = Draw
            current_pos = lmList[8][:2]
            if self.prev_pos is None:
                self.prev_pos = current_pos
            cv2.line(self.canvas, self.prev_pos, current_pos, (255, 0, 255), 10)
            self.prev_pos = current_pos

        elif fingers == [1, 0, 0, 0, 0]:  # Thumb up = Clear
            self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.prev_pos = None

        elif fingers == [0, 0, 0, 0, 1]:  # Pinky up = Submit to AI
            pil_image = Image.fromarray(self.canvas)
            try:
                response = model.generate_content(["Solve this math problem", pil_image])
                global output_global
                output_global = response.text
            except Exception as e:
                output_global = f"Error: {e}"

        else:
            self.prev_pos = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global canvas_global

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        canvas_resized = cv2.resize(self.canvas, (img.shape[1], img.shape[0]))

        hands, _ = self.detector.findHands(img, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = self.detector.fingersUp(hand)
            self.draw(fingers, lmList)

        overlay = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)
        canvas_global = overlay
        return av.VideoFrame.from_ndarray(overlay, format="bgr24")


if run:
    webrtc_streamer(
        key="math-gesture",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

if output_global:
    output_text_area.markdown(f"**{output_global}**")
