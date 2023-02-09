from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

# use_gpu = True
# live_video = False
confidence_level = 0.5
fps = FPS().start()
object_name=[]
accuracy=[]
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor",]
path='sample.mp4'
def object_detection(path):
    ret = True
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # print(len(COLORS))    

    net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

    # if use_gpu:
    #     print("[INFO] setting preferable backend and target to CUDA...")
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
 
    vs = cv2.VideoCapture(path)
    print(vs)
    while ret:
        ret, frame = vs.read()
        if ret:
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_level:
                    idx = int(detections[0, 0, i, 1])
                    print("THE object name is ",CLASSES[idx],"  and accuracy found is :",confidence*100)
                    object_name.append(CLASSES[idx])
                    accuracy.append(confidence*100)
    # print(len(object_name),len(accuracy))
    return object_name,accuracy
def main():
    object_name,accuracy=object_detection(path)
if __name__=="__main__":
    main()