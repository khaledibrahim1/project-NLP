import cv2
import numpy as np
import face_recognition
import os
import requests
from bs4 import BeautifulSoup

path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print("Detected persons:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('✅ Encodings generated successfully.')

cap = cv2.VideoCapture(0)

recognized_name = None

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocations = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceLocations)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceLocations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].title()
            recognized_name = name
            print(f"✅ Recognized: {name}")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cap.release()
            cv2.destroyAllWindows()
            break

    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) == ord('q') or recognized_name:
        break
#===================================================
def get_player_info(name):
    print(f"\n🔍 Gathering data about: {name} from Wikipedia...\n")
    wiki_name = name.replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/{wiki_name}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching the page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    infobox = soup.find('table', {'class': 'infobox'})

    if not infobox:
        print("❌ No information found.")
        return

    for row in infobox.find_all('tr'):
        header = row.find('th')
        data = row.find('td')
        if header and data:
            key = header.text.strip()
            value = data.text.strip()
            print(f"{key}: {value}")

if recognized_name:
    get_player_info(recognized_name)
else:
    print("❗ No face recognized.")