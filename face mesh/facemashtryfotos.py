import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

file_name = "lip_outer_x.txt"
file_name2 = "lip_inner_x.txt"
file_name3 = "lip_outer_y.txt"
file_name4 = "lip_inner_y.txt"

lipsUpperOuter = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409])
lipsLowerOuter = np.array([146, 91, 181, 84, 17, 314, 405, 321, 375, 291])
lipsUpperInner = np.array([78, 191, 80, 81, 82, 13, 312, 311, 310, 415])
lipsLowerInner = np.array([308, 95, 88, 178, 87, 14, 317, 402, 318, 324])

def save_landmarks(file_name, landmarks, length):
    with open(file_name, 'a') as file:
        for i in range(0, len(landmarks), length):
            line = ','.join(map(str, landmarks[i:i+length]))
            file.write(line + '\n')

degisken = 4

while True:
    photos = f"C:\\Users\\yaman\\Desktop\\2209\\dataset\\{degisken}.jpg"
    image = cv2.imread(photos)
    if image is None:
        break 
    
    lipsUpperOuter_tempory_x = np.array([], dtype=np.int32)
    lipsLowerOuter_tempory_x = np.array([], dtype=np.int32)
    lipsUpperInner_tempory_x = np.array([], dtype=np.int32)
    lipsLowerInner_tempory_x = np.array([], dtype=np.int32)

    lipsUpperOuter_tempory_y = np.array([], dtype=np.int32)
    lipsLowerOuter_tempory_y = np.array([], dtype=np.int32)
    lipsUpperInner_tempory_y = np.aqrray([], dtype=np.int32)
    lipsLowerInner_tempory_y = np.array([], dtype=np.int32)

    image = cv2.flip(image, 1)
    image = cv2.resize(image, (640, 480))

    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                if idx in lipsLowerInner:
                    lipsLowerInner_tempory_x = np.append(lipsLowerInner_tempory_x, x)
                    lipsLowerInner_tempory_y = np.append(lipsLowerInner_tempory_y, y)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                if idx in lipsUpperInner:
                    lipsUpperInner_tempory_x = np.append(lipsUpperInner_tempory_x, x)
                    lipsUpperInner_tempory_y = np.append(lipsUpperInner_tempory_y, y)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                if idx in lipsLowerOuter:
                    lipsLowerOuter_tempory_x = np.append(lipsLowerOuter_tempory_x, x)
                    lipsLowerOuter_tempory_y = np.append(lipsLowerOuter_tempory_y, y)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                if idx in lipsUpperOuter:
                    lipsUpperOuter_tempory_x = np.append(lipsUpperOuter_tempory_x, x)
                    lipsUpperOuter_tempory_y = np.append(lipsUpperOuter_tempory_y, y)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        if len(lipsUpperOuter_tempory_x) > 0 and len(lipsLowerOuter_tempory_x) > 0:
            lipssouter_x = lipsUpperOuter_tempory_x[:10] - lipsLowerOuter_tempory_x[:10]
            save_landmarks(file_name, lipssouter_x, 10)
        if len(lipsUpperInner_tempory_x) > 0 and len(lipsLowerInner_tempory_x) > 0:
            lipssinner_x = lipsUpperInner_tempory_x[:10] - lipsLowerInner_tempory_x[:10]
            save_landmarks(file_name2, lipssinner_x, 10)
        if len(lipsUpperOuter_tempory_y) > 0 and len(lipsLowerOuter_tempory_y) > 0:
            lipssouter_y = lipsUpperOuter_tempory_y[:10] - lipsLowerOuter_tempory_y[:10]
            save_landmarks(file_name3, lipssouter_y, 10)
        if len(lipsUpperInner_tempory_y) > 0 and len(lipsLowerInner_tempory_y) > 0:
            lipssinner_y = lipsUpperInner_tempory_y[:10] - lipsLowerInner_tempory_y[:10]
            save_landmarks(file_name4, lipssinner_y, 10)

        cv2.imshow("Image", image)

    degisken += 1

    if cv2.waitKey(3000) & 0xFF == ord('q'):
        continue

cv2.destroyAllWindows()
