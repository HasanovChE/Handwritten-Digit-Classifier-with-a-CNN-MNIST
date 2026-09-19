# app_streamlit.py
import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="centered"
)

st.title("MNIST Digit Classifier (CNN)")
st.write("Upload an image of handwriting and see how our model recognizes it.")

uploaded_file = st.file_uploader(
    "Select the image file (PNG, JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)
    
    if st.button("Make a Prediction", type="primary"):
        with st.spinner("The model is analyzing..."):
            files = {"file": uploaded_file.getvalue()}
            
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )
                
                if response.status_code == 200:
                    data = response.json()
                    digit = data["digit"]
                    confidence = data["confidence"]
                    
                    st.success(f"Projected Figure: **{digit}**")
                    st.info(f"Level of Confidence: **{confidence:.2%}**")
                else:
                    st.error("A server error occurred!")
                    
            except requests.exceptions.ConnectionError:
                st.error(
                    "It was not possible to connect to the FastAPI server. "
                    "Please start the server first using the `uvicorn api.app:app` command!"
                )