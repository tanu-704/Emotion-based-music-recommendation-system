import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

# ----------------------------
# Load Model & Labels
# ----------------------------
model = load_model("model.h5", compile=False)
label = np.load("labels.npy")

# ----------------------------
# Mediapipe Setup
# ----------------------------
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

holis = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.header("🎵 Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = True

# ----------------------------
# Load Emotion Safely
# ----------------------------
emotion = ""
if os.path.exists("emotion.npy"):
    try:
        emotion = np.load("emotion.npy", allow_pickle=True)[0]
    except:
        emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

# ----------------------------
# Emotion Processor Class
# ----------------------------
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        # ---------------- FACE ----------------
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
        else:
            lst.extend([0.0] * 468 * 2)

        # ---------------- LEFT HAND ----------------
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 21 * 2)

        # ---------------- RIGHT HAND ----------------
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 21 * 2)

        lst = np.array(lst).reshape(1, -1)

        # ---------------- PREDICTION ----------------
        pred = label[np.argmax(model.predict(lst, verbose=0))]

        # Save emotion
        np.save("emotion.npy", np.array([pred]))

        # Show text on frame
        cv2.putText(frm, str(pred), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        # ---------------- DRAW LANDMARKS ----------------
        if res.face_landmarks:
            drawing.draw_landmarks(
                frm,
                res.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                drawing.DrawingSpec(thickness=1)
            )

        if res.left_hand_landmarks:
            drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if res.right_hand_landmarks:
            drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# ----------------------------
# Inputs
# ----------------------------
lang = st.text_input("Language")
singer = st.text_input("Singer")

# ----------------------------
# Start Camera
# ----------------------------
if lang and singer and st.session_state["run"]:
    webrtc_streamer(
        key="emotion-app",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor
    )

# ----------------------------
# Recommendation Button
# ----------------------------
btn = st.button("Recommend Me Songs 🎶")

if btn:
    if not emotion:
        st.warning("Please allow camera to detect your emotion first.")
        st.session_state["run"] = True
    else:
        query = f"{lang}+{emotion}+song+{singer}"
        url = f"https://www.youtube.com/results?search_query={query}"

        st.success("Opening YouTube recommendations...")
        st.markdown(f"[👉 Click here if not opened]({url})")

        # Optional fallback
        webbrowser.open(url)

        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False
		
