import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image

st.set_option("deprecation.showfileUploaderEncoding", False)

# Load dlib's face landmarks detector
predictor_path = "https://github.com/Franky1/Face-Averaging-App/blob/main/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Calculate the face aesthetic index
def calculate_face_aesthetic_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None, "No faces detected"

    face = faces[0]
    landmarks = predictor(gray, face)

    # Calculate the aesthetic index
    face_width = np.sqrt((landmarks.part(16).x - landmarks.part(0).x)**2 + (landmarks.part(16).y - landmarks.part(0).y)**2)
    face_height = np.sqrt((landmarks.part(8).x - landmarks.part(27).x)**2 + (landmarks.part(8).y - landmarks.part(27).y)**2)
    aesthetic_index = face_height / face_width

    return aesthetic_index, None

def main():
    st.title("Face Aesthetic Index Calculator")
    uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = np.array(image)
        aesthetic_index, error = calculate_face_aesthetic_index(image)

        if error:
            st.error(error)
        else:
            st.subheader(f"Face Aesthetic Index: {aesthetic_index:.2f}")

if __name__ == "__main__":
    main()
