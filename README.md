# ✍️ Math Gesture Recognition

A real-time gesture-based math expression interpreter that allows users to draw mathematical equations in the air using their fingers. The system tracks hand movements using a webcam and recognizes finger gestures to draw, clear, and submit math problems. It sends the captured drawing to Google’s Gemini AI, which interprets and solves the expression and returns the result on a live Streamlit dashboard.

---

## 📌 Features

- ✋ Real-time hand tracking using a webcam
- ✍️ Draw mathematical expressions with just your index finger
- 🧽 Clear the canvas with a thumb gesture
- 📤 Submit the expression using a pinky gesture
- 💬 Output the solution directly on the desktop using Streamlit
- ⚡ Uses Google's Gemini 1.5 Flash AI for interpreting handwritten math

---

## 🛠️ Tech Stack

| Area           | Technology                    |
|----------------|-------------------------------|
| Hand Tracking  | OpenCV, cvzone, mediapipe     |
| UI             | Streamlit                     |
| AI/ML Model    | Google Generative AI (Gemini) |
| Image Handling | Pillow, NumPy                 |
| Programming    | Python                        |

---

## 🚀 How It Works

1. Webcam captures the hand gestures in real time.
2. If the index finger is up, it draws on the canvas.
3. If the thumb is up, the canvas is cleared.
4. If the pinky is up, the canvas is captured and sent to Gemini for processing.
5. Gemini responds with the solution which is displayed on the Streamlit app.

---

## 🧪 How to Use

1. Ensure you have a webcam connected.
2. Activate your Python environment.
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Set your Google API key securely as an environment variable:  
   On Windows (CMD):  
   ```bash
   set GOOGLE_API_KEY=your_api_key_here
   ```
   On Linux/macOS:  
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```
5. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

---

## 🔧 How to Run

1. Clone or download the project folder.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install requirements and run as explained above.

---

## 👩‍💻 About Me

Hi, I’m **Jhansi Challa**, a passionate developer who enjoys building intelligent interfaces and AI-powered applications. I enjoy working with modern front-end stacks, experimenting with AI-driven tools, and solving real-world problems through code. 

Let's connect on [LinkedIn](linkedin.com/in/challajhansi) or reach out via email at: **jhansichalla@26gmail.com**

---

© 2025 Jhansi Challa