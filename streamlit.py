import streamlit as st
st.title("CoNR Test")

video_file = open("qwq.mp4", "rb").read()
st.video(video_file)
