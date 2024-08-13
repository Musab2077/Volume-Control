import time 
import cv2
import mediapipe as mp
import numpy as np
import hand_tracking_module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

w_cam,h_cam=640,480

cap=cv2.VideoCapture(0)
cap.set(3,w_cam)
cap.set(3,h_cam)
detector=htm.HandDetetor(detection_conf=0.7)
p_time=0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

min_volume=vol_range[0]
max_volume=vol_range[1]

while cap.isOpened():
    success,frame=cap.read()
    frame=detector.hands_detection(frame)
    landmarks = detector.hand_landmark(frame,0)
    if len(landmarks)!=0:
        # print(landmarks[4],landmarks[8])

        x4,y4=landmarks[4][1],landmarks[4][2]
        x8,y8=landmarks[8][1],landmarks[8][2]

        cv2.circle(frame,(x4,y4),10,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(x8,y8),10,(255,0,0),cv2.FILLED)

        mid_x,mid_y=(x4+x8)//2,(y4+y8)//2

        cv2.line(frame,(x4,y4),(x8,y8),(0,0,0),8)

        line_length=math.hypot(x8-x4,y8-y4)
        print(line_length)
        
        cv2.circle(frame,(mid_x,mid_y),10,(0,0,255),cv2.FILLED)
        
        vol=np.interp(line_length,[50,300],[min_volume,max_volume])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if line_length<50:
            cv2.circle(frame,(mid_x,mid_y),10,(0,255,0),cv2.FILLED)

    c_time=time.time()
    fps=1/(c_time-p_time)
    p_time=c_time

    cv2.putText(frame,f'FPS: {int(fps)}',(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),5)

    cv2.imshow('Volume Control Project',frame)
    
    if cv2.waitKey(1) & 0XFF==ord('q'):
        # print(len(landmarks))
        break

cap.release()
cv2.destroyAllWindows()
