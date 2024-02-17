from pathlib import Path
import streamlit as st
import app
import app_jetson
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Driver Drowsiness Detection")

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select source:", ["Webcam"])

if source_radio == "Webcam":
    #If it is an Ubuntu 22.04 laptop:
    app.play_webcam()

    #If it is a Jetson Nano, uncomment the following line:
    # app_jetson.play_webcam()