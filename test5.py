from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3 as ts
import time
from playsound import playsound
import os
# import pyttsx
import matplotlib as plt

use_gpu = True
live_video = False
confidence_level = 0.5
fps = FPS().start()
ret = True
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor",]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')
object_size = 1
focal_length = 800.0 

print("[INFO] accessing video stream...")

if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture('test1.mp4')
object_name=dict()
accuracy=[]
while ret:
    ret, frame = vs.read()
    if ret:
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        # print(blob)
        net.setInput(blob)
        detections = net.forward()
        direction="unknown"
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # print(startX,startY,endX,endY)
                
                # print(frame.shape)   
                xavg=(startX+endX)/2
                yavg=(startY+endY)/2
                if xavg>200 and yavg>112.5:
                    direction="topright"
                elif xavg>200 and yavg<112.5:
                    direction="downright"
                elif xavg<200 and yavg>112.5:
                    direction="topleft"
                elif xavg<200 and yavg<112.5:
                    direction="downleft"
                object_pixels = max(w, h)

                distance = (object_size * focal_length) / object_pixels 
                label = "{}: {:.2f}%  {} {} ".format(CLASSES[idx], confidence * 100,direction,distance)
                text="THE object name is {}:accuracy found is {:.2f}% {} {} ".format(CLASSES[idx], confidence * 100,direction,distance)
                # language = 'en'
                # audio = gTTS(text=text, lang=language, slow=False)
                # audio.save("x.mp3")
                # playsound("x.mp3")
                # time.sleep(0.5)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
        frame = imutils.resize(frame,width=400)
        cv2.imshow('Live detection',frame)
        # print(frame.shape)
        if cv2.waitKey(1)==27:
            break

        fps.update()
fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# print(len(object_name))
