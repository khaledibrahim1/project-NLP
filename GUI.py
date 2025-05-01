import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .player-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        background-color: white;
    }
    .player-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Face Recognition System")
    st.markdown("---")
    st.subheader("Configuration")
    detection_confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.6, 0.05)
    st.markdown("---")
    st.info("This system recognizes faces and retrieves information from Wikipedia.")

# Main content
st.title("👤 Face Recognition and Information System")
st.markdown("---")

# Initialize session state
if 'recognized_name' not in st.session_state:
    st.session_state.recognized_name = None
if 'player_info' not in st.session_state:
    st.session_state.player_info = None
if 'run_recognition' not in st.session_state:
    st.session_state.run_recognition = False

# Load known faces
@st.cache_resource
def load_known_faces():
    path = 'persons'
    images = []
    classNames = []
    personsList = os.listdir(path)

    for cl in personsList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
    
    # Generate encodings
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList, classNames

try:
    encodeListKnown, classNames = load_known_faces()
    st.sidebar.success(f"Loaded {len(classNames)} known faces")
except Exception as e:
    st.sidebar.error(f"Error loading known faces: {str(e)}")

# Face recognition function
def recognize_faces():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Recognition")
    
    while st.session_state.run_recognition and not stop_button:
        success, img = cap.read()
        if not success:
            st.error("Failed to capture image from camera")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceLocations = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceLocations)

        for encodeFace, faceLoc in zip(encodesCurFrame, faceLocations):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < (1 - detection_confidence):
                name = classNames[matchIndex].title()
                st.session_state.recognized_name = name
                
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                st.session_state.run_recognition = False
                break

        frame_placeholder.image(img, channels="BGR", use_column_width=True)
        
        if cv2.waitKey(1) == ord('q') or stop_button:
            st.session_state.run_recognition = False
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Wikipedia scraping function
def get_player_info(name):
    wiki_name = name.replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/{wiki_name}"
    info_dict = {}

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        infobox = soup.find('table', {'class': 'infobox'})

        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                data = row.find('td')
                if header and data:
                    key = header.text.strip()
                    value = data.text.strip().replace('\n', ' ')
                    info_dict[key] = value
            
            # Try to get image
            image = infobox.find('img')
            if image and 'src' in image.attrs:
                image_url = 'https:' + image['src']
                info_dict['image_url'] = image_url
        
        st.session_state.player_info = info_dict
    except Exception as e:
        st.error(f"Error fetching information: {str(e)}")

# Start recognition button
if st.button("Start Face Recognition") and not st.session_state.run_recognition:
    st.session_state.run_recognition = True
    recognize_faces()

# Display recognition results
if st.session_state.recognized_name:
    st.markdown(f"""
    <div class="success-box">
        <h3>✅ Recognized: {st.session_state.recognized_name}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Get Player Information"):
        with st.spinner(f"Fetching information about {st.session_state.recognized_name}..."):
            get_player_info(st.session_state.recognized_name)

# Display player information
if st.session_state.player_info:
    info = st.session_state.player_info
    st.markdown(f"""
    <div class="player-card">
        <h2>{st.session_state.recognized_name}</h2>
    """, unsafe_allow_html=True)
    
    if 'image_url' in info:
        st.image(info['image_url'], width=200)
    
    cols = st.columns(2)
    col_idx = 0
    
    for key, value in info.items():
        if key != 'image_url':
            with cols[col_idx % 2]:
                st.markdown(f"""
                <div class="info-box">
                    <strong>{key}:</strong> {value}
                </div>
                """, unsafe_allow_html=True)
            col_idx += 1
    
    st.markdown("</div>", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ How to use this system"):
    st.markdown("""
    1. Make sure you have a webcam connected
    2. Click 'Start Face Recognition' to begin
    3. The system will attempt to recognize faces from the database
    4. Once recognized, click 'Get Player Information' to retrieve details
    5. Use 'Stop Recognition' to stop the camera at any time
    
    **Note:** The system only recognizes faces that are in the 'persons' folder.
    """)

# Known faces section
with st.expander("📁 Known Faces in Database"):
    if classNames:
        st.write("The system can recognize the following faces:")
        for name in classNames:
            st.write(f"- {name.title()}")
    else:
        st.warning("No known faces found in the database.")