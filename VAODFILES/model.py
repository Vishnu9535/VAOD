from imutils.video import FPS
import cv2
import imutils  
import numpy as np

objectlist = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

COLORS = np.random.uniform(0, 255, size=(len(objectlist), 3))
load_model = cv2.dnn.readNetFromCaffe('modelfiles/cnnmodule.prototxt', 'modelfiles/Mobilenetssd.model')
focal_length = 0.026   
confidence_level = 0.5    
def final_model():
        """
        It detects objects using the deep learning model
        Gives the video  with object detections and gives audio output for each frame of the video.
        """
        fps = FPS().start()
        status = True
   
        video_obj = cv2.VideoCapture('videos/test1.mp4')

        while status:
            status, frame = video_obj.read()
            if status:
                frame = imutils.resize(frame, width=400)
                (h, w) = frame.shape[:2]

                image_feature = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                load_model.setInput(image_feature)
                detect = load_model.forward()

                for i in np.arange(0, detect.shape[2]):
                    confidence = detect[0, 0, i, 2]
                    if confidence > confidence_level:
                        class_of_object = int(detect[0, 0, i, 1])
                        bounding_box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (topleft_x, topleft_y, downright_x, downright_y) = bounding_box.astype("int")

                        label = f"{objectlist[class_of_object]} {confidence * 100:.2f}% "
                        cv2.rectangle(frame, (topleft_x, topleft_y), (downright_x, downright_y), COLORS[class_of_object], 2)
                        y = topleft_y - 15 if topleft_y - 15 > 15 else topleft_y + 15
                        cv2.putText(frame, label, (topleft_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_of_object], 2)

                frame = imutils.resize(frame, width=400)
                cv2.imshow('Live detection', frame)

                if cv2.waitKey(1) == 5:
                    break

                fps.update()

        fps.stop()
final_model()