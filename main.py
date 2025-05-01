import cv2
import numpy as np
import face_recognition
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f'face_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_known_faces(path='persons'):
    """Load known faces from the specified directory"""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
            return [], []

        images = []
        classNames = []
        personsList = os.listdir(path)
        
        for cl in personsList:
            try:
                curPerson = cv2.imread(f'{path}/{cl}')
                if curPerson is not None:
                    images.append(curPerson)
                    classNames.append(os.path.splitext(cl)[0])
                    logging.info(f"Successfully loaded image: {cl}")
                else:
                    logging.error(f"Failed to load image: {cl}")
            except Exception as e:
                logging.error(f"Error processing image {cl}: {str(e)}")
        
        return images, classNames
    except Exception as e:
        logging.error(f"Error in load_known_faces: {str(e)}")
        return [], []

def find_encodings(images):
    """Generate face encodings for the given images"""
    encodeList = []
    try:
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        logging.info(f"Successfully encoded {len(encodeList)} faces")
    except Exception as e:
        logging.error(f"Error in find_encodings: {str(e)}")
    return encodeList

def main():
    try:
        # Load known faces
        images, classNames = load_known_faces()
        if not images:
            logging.warning("No known faces found in the persons directory")
            return

        # Generate encodings
        encodeListKnown = find_encodings(images)
        if not encodeListKnown:
            logging.error("Failed to generate face encodings")
            return

        logging.info("Face recognition system initialized successfully")

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return

        while True:
            success, img = cap.read()
            if not success:
                logging.error("Failed to capture frame from camera")
                break

            # Resize image for faster processing
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Find faces in the current frame
            faceCurrentFrame = face_recognition.face_locations(imgS)
            encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

            for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    logging.info(f"Recognized face: {name}")
                    
                    # Scale back up face locations
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    
                    # Draw rectangle around face
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            cv2.imshow('Face Recognition', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        logging.info("Face recognition system shut down")

if __name__ == "__main__":
    main()