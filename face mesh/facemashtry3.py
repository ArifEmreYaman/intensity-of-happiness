import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)


file_name = "lip_outer.txt"
file_name2 = "lip_inner.txt"

lipsUpperOuter = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409])
lipsLowerOuter = np.array([146, 91, 181, 84, 17, 314, 405, 321, 375, 291])
lipsUpperInner = np.array([78, 191, 80, 81, 82, 13, 312, 311, 310, 415])
lipsLowerInner = np.array([308, 95, 88, 178, 87, 14, 317, 402, 318, 324])


def save_landmarks(file_name, landmarks, length):
    with open(file_name, 'w') as file:
        for i in range(0, len(landmarks), length):
            line = ','.join(map(str, landmarks[i:i+length]))
            file.write(line + '\n')


lipsUpperOuter_tempory = np.array([], dtype=np.int32)
lipsLowerOuter_tempory = np.array([], dtype=np.int32)
lipsUpperInner_tempory = np.array([], dtype=np.int32)
lipsLowerInner_tempory = np.array([], dtype=np.int32)
while cap.isOpened():
    _, image = cap.read()
    if not _:
        print("KAMERA GÖRÜNTÜ VERMİYOOOOORRRRRRR.")
        break
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
  
    try:
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    if idx in lipsLowerInner:
                        lipsLowerInner_tempory = np.append(lipsLowerInner_tempory, [x, y])
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                    if idx in lipsUpperInner:
                        lipsUpperInner_tempory = np.append(lipsUpperInner_tempory, [x, y])
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                    if idx in lipsLowerOuter:
                        lipsLowerOuter_tempory = np.append(lipsLowerOuter_tempory, [x, y])
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                    if idx in lipsUpperOuter:
                        lipsUpperOuter_tempory = np.append(lipsUpperOuter_tempory, [x, y])
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            # En küçük uzunluğu belirleyerek dizileri aynı boyuta getirin
            # Matematiksel işlemleri yapın
            # min_length = min(len(lipsUpperOuter_tempory), len(lipsLowerOuter_tempory), 
            #                  len(lipsUpperInner_tempory), len(lipsLowerInner_tempory))
            if len(lipsUpperOuter_tempory) > 0 and len(lipsLowerOuter_tempory) > 0:
                lipssouter = lipsUpperOuter_tempory - lipsLowerOuter_tempory
                save_landmarks(file_name, lipssouter, 10)
            if len(lipsUpperInner_tempory) > 0 and len(lipsLowerInner_tempory) > 0:
                lipssinner = lipsUpperInner_tempory - lipsLowerInner_tempory
                save_landmarks(file_name2, lipssinner, 11)
        cv2.imshow('Face Mesh', image)
    except Exception as e:
        print(f"Hata oluştu: {e}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()