from djitellopy import tello
import time
import cv2
import numpy as np

class drone_handler:
    def __init__(self, HAVE_DRONE):
        self.HAVE_DRONE = HAVE_DRONE
        self.updown_speed = 30
        if self.HAVE_DRONE:
            self.DRONE = tello.Tello()
            self.DRONE.connect()
            print(self.DRONE.get_battery())
            self.DRONE.streamoff()
            self.DRONE.streamon()
            img = self.DRONE.get_frame_read().frame
            img = self.DRONE.get_frame_read().frame
            img = self.DRONE.get_frame_read().frame

            print("takeoff")
            self.DRONE.takeoff()

    def end(self):
        if self.HAVE_DRONE:
            self.DRONE.end()
        else:
            print("end drone")

    def get_height(self):
        if self.HAVE_DRONE:
            return self.DRONE.get_current_state()['tof']
        else:
            return 50

    def get_battery(self):
        if self.HAVE_DRONE:
            return self.DRONE.get_battery()
        else:
            return 100

    def get_img(self):
        if self.HAVE_DRONE:
            try:
                img = self.DRONE.get_frame_read().frame
            except:
                img = np.zeros((600, 800, 3), np.uint8)
        else:
            img = np.zeros((600, 800, 3), np.uint8)
        return img
        
    def get_drone_state(self):
        return self.DRONE.get_current_state()

    def send_command(self, lr,fb,ud,yv):
        lr = int(lr)
        fb = int(fb)
        ud = int(ud)
        yv = int(yv)
        if self.HAVE_DRONE:
            self.DRONE.send_rc_control(lr, fb,ud,yv)
        else:
            #print((lr,fb,ud,yv))
            pass

    def flip(self):
        if self.HAVE_DRONE:
            self.DRONE.flip_forward()
        else:
            print("flip")

    def land(self):
        if self.HAVE_DRONE:
            self.DRONE.land()
        else:
            print("land")