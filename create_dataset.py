import cv2
import mediapipe as mp
import numpy as np
import csv
from control_drone import drone_handler

with open("gesture_train.csv", 'r', encoding='utf-8') as load_file:
    data =  load_file.readlines()

while True:
    try:
        data.remove("\n")

    except:
        break

with open("gesture_train.csv", 'w', encoding='utf-8') as save_file:
    for line in data:
        save_file.write(line)
        save_file.write("\n")

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img  =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
                if j == 0:
                    dx1 = int(lm.x*img.shape[0])
                    dy1 = int(lm.y*img.shape[1])
                    
                elif j == 12:
                    dx2 = int(lm.x*img.shape[0])
                    dy2 = int(lm.y*img.shape[1])
                    
            v1 = joint[[0,1,2,3,0,5,6,7,0,9, 10,11,0, 13,14,15,0, 17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]

            v = v2 - v1 
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9, 10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

            angle = np.degrees(angle) 

            data = np.array([angle], dtype=np.float32)
            
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('create_data-set', img)
    key_input = cv2.waitKey(1)
    save_data = True
    if key_input == ord('q'):
        break
    else:
        if key_input == 49: #키보드 숫자 1키
            gesture_number = '0.000000'
        elif key_input == 50: #키보드 숫자 2키
            gesture_number = '1.000000'
        elif key_input == 51: #키보드 숫자 3키
            gesture_number = '2.000000'
        elif key_input == 52: #키보드 숫자 4키
            gesture_number = '3.000000'
        elif key_input == 53: #키보드 숫자 5키
            gesture_number = '4.000000' 
        elif key_input == 54: #키보드 숫자 6키
            gesture_number = '5.000000'
        elif key_input == 55: #키보드 숫자 7키
            gesture_number = '6.000000'
        else:
            save_data = False
        if save_data:
            with open("gesture_train.csv", "a") as f:
                writer = csv.writer(f)
                data = list(data[0])
                data.append(gesture_number)
                writer.writerow(data)
                print("d")