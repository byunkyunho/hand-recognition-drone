import cv2
import mediapipe as mp
import numpy as np
import csv
from control_drone import drone_handler
import time
import sys
from datetime import datetime

image_size = 1

def get_hand_gesture(res):
    global dx1, dy1, dx2, dy2
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
    ret, results, neighbours, dist = knn.findNearest(data, k=3)
    idx = int(results[0][0])
    return idx, results

def organize_data(most_left, most_up, most_right, most_down):
    most_left = int(most_left*640) - 10
    most_up = int(most_up*480) - 10
    most_down = int(most_down*480) + 10
    most_right = int(most_right*640) +10
    if most_left < 1:
        most_left = 0
    if most_right > 639:
        most_right = 640
    if most_up < 1:
        most_up = 0     
    if most_down > 479:
        most_down = 480    
    return most_left, most_up, most_right, most_down

def get_motion(recent_gesture):
    recent_gesture_string = " ".join(recent_gesture) 
    for a in range(2,6):
        if "open " + "back "*a+"open" in recent_gesture_string:
            return 'flip'
    for a in range(2,9):
        if "open " + "feast "*a+"open" in recent_gesture_string:
            return 'cap'      

def mosaic_image(img, x1,y1,x2,y2):
    rate = 15
    cut_image = img[y1:y2, x1:x2] 
    cut_image = cv2.resize(cut_image, (int(int(abs(x2-x1))//rate), int(int(abs(y1-y2))//rate)))
    cut_image = cv2.resize(cut_image, (int(abs(x2-x1)),int(abs(y1-y2))), interpolation=cv2.INTER_AREA)
    return cut_image

def get_hand_rect(hand_landmark):
    ml = 1000
    mr = 0
    mu = 1000
    md = 0
    for landmark in hand_landmark:
        if landmark.x > mr:
            mr = landmark.x
        if landmark.x < ml:
            ml = landmark.x
        if landmark.y > md:
            md = landmark.y
        if landmark.y < mu:
            mu = landmark.y   
    return ml, mu, mr, md

def process_image(img):
    img = cv2.flip(img, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    result = hands.process(img)

    img  =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img , result

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )

file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

if len(sys.argv) > 1:
    if sys.argv[1] == '1':
        DRONE = drone_handler(True)
    else:
        DRONE = drone_handler(False)
else:
    DRONE = drone_handler(False)

recent_gesture = ['open']

view_special_message = 0

running = True            

save_image = False

while running:
    LEFT_STOP = False
    RIGHT_STOP = False  
    lr = fb = ud = yv = 0
    img,result = process_image(cap.read()[1])
    drone_height = DRONE.get_height()

    hand_list = {}
    if result.multi_hand_landmarks is not None: 
        for i,hand in enumerate(result.multi_handedness):
            hand_list[hand.classification[0].label] = i

        if 'Left' in hand_list.keys():
            res = result.multi_hand_landmarks[hand_list['Left']]
            idx, results = get_hand_gesture(res)

            message = None

            if idx == 0:
                if res.landmark[4].x > res.landmark[17].x and not (dx1-dx2) == 0:
                    if abs((dy1-dy2)/(dx1-dx2)) > 4:
                        message = "open"
                    elif  abs((dy1-dy2)/(dx1-dx2))  < 3:
                        if dx1 > dx2:
                            message = "left turn"
                            yv = -50
                        elif dx1 < dx2:
                            message = "right turn"
                            yv = 50

                else:
                    message = "back"

            elif idx == 1:
                message = "stop"
                LEFT_STOP = True
            
            elif idx == 3 or idx == 2:
                message = 'feast'

            if idx == 4:
                left_fuck = True
                most_left, most_up, most_right, most_down = get_hand_rect(res.landmark)
                most_left, most_up, most_right, most_down = organize_data(most_left, most_up, most_right, most_down)

                img[most_up:most_down, most_left:most_right] = mosaic_image(img,most_left, most_up, most_right, most_down)
            else:
                left_fuck = False
            
            if message:
                if len(recent_gesture) < 12:
                    recent_gesture.append(message)
                else:
                    for i in range(11):
                        recent_gesture[i] = recent_gesture[i+1]
                    recent_gesture[11] = message

            recent_motion = get_motion(recent_gesture)
            #print(recent_gesture)
            if recent_motion == "flip":
                recent_gesture = ['open']
                message = 'flip'
                special_message = 'flip'
                view_special_message = 20
                if DRONE.get_battery() > 55:
                    print(flip_direction)
                    DRONE.flip()

            elif recent_motion == 'cap':
                recent_gesture = ['open']
                message = 'cap'
                view_special_message = 20
                special_message = 'cap'
                save_image = True
                print("cap")
            
            if  view_special_message > 0:
                cv2.putText(img, text=special_message, org=(dx1+40, dy1-90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
                view_special_message-=1

            elif message == 'right turn' or message == 'left turn' or message == 'stop':
                cv2.putText(img, text=message, org=(dx1+10, dy1-85), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

            if not left_fuck:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        flip_direction = 'f'

        if 'Right' in hand_list.keys():
            res = result.multi_hand_landmarks[hand_list['Right']]

            idx  = None

            idx, results = get_hand_gesture(res)

            message = None
            
            if idx == 0:
                if (dx1-dx2) != 0:
                    if abs((dy1-dy2)/(dx1-dx2)) > 4:
                        if res.landmark[4].x < res.landmark[17].x:
                            message = "forward"
                            speed = (res.landmark[0].z - res.landmark[12].z) * 1000 if (res.landmark[0].z - res.landmark[12].z) * 1000 > 1 else 0
                            fb = speed
                        else:
                            speed = (res.landmark[9].z - res.landmark[0].z) * 1000 if (res.landmark[9].z - res.landmark[0].z) * 1000 > 1 else 0
                            speed = 50
                            message = "backward"
                            fb = -speed 
                    elif  abs((dy1-dy2)/(dx1-dx2))  < 3.5:
                        speed = (100 - int((abs((dy1-dy2)/(dx1-dx2)) * 50))) if (100 - int((abs((dy1-dy2)/(dx1-dx2)) * 50))) > 1 else 0
                        if dx1 > dx2:
                            message = "left"
                            lr = -speed
                        elif dx1 < dx2:
                            message = "right"
                            speed = int(speed*1.3)
                            lr = speed

                    if message:
                        cv2.putText(img, text="speed : "+str(int((speed))), org=(dx1+40, dy1-60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(0, 255, 0), thickness=2)

            elif idx == 1:
                message = "stop"
                RIGHT_STOP = True

            elif idx == 2:
                if res.landmark[4].y < res.landmark[17].y:
                    message = "up"
                    ud = 30
                else:
                    message = "down"
                    ud = -30

            if idx == 4:
                right_fuck = True
                most_left, most_up, most_right, most_down = get_hand_rect(res.landmark)
                most_left, most_up, most_right, most_down = organize_data(most_left, most_up, most_right, most_down)
                img[most_up:most_down, most_left:most_right] = mosaic_image(img,most_left, most_up, most_right, most_down )
            
                img = cv2.rectangle(img, (int(most_left*640), int(most_up*480)),(int(most_right*640), int(most_down*480)), (0,255,0),3)
            else:
                right_fuck = False

            if message:
                cv2.putText(img, text=message, org=(dx1+40, dy1-90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            
            if not right_fuck:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    if drone_height < 30:
        running = False
        DRONE.land()
        print("land")
        break
    if RIGHT_STOP:
        lr = fb = ud = 0
    if LEFT_STOP:
        yv = 0
        
    DRONE.send_command(lr,fb,ud,yv)

    cv2.putText(img, text=str(DRONE.get_battery()), org=(560,40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

    drone_img = cv2.resize(DRONE.get_img(), (int(img.shape[1]*image_size) , int(img.shape[0]*image_size)))
    img =  cv2.resize(img, (int(img.shape[1]*image_size) , int(img.shape[0]*image_size)))
    
    show_img = cv2.hconcat([drone_img, img])

    if save_image:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        print(now)
        cv2.imwrite(f'image/{now}.png', show_img)
        save_image = False
        
    cv2.imshow('main img', show_img)

    key_input = cv2.waitKey(1)

    if key_input == ord('q'):
        DRONE.land()
        running = False
        break
    
cv2.destroyAllWindows()

DRONE.end()
