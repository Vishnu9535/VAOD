from imutils.video import FPS
from object_size import calculate_object_size
import cv2
import imutils  
import math
import numpy as np
from audio import audio_output
from direction import get_direction
from distance import calculate_distance

class ObjectDetection:
    def __init__(self):
        """
        Initialize the ObjectDetection class.
        """
        # self.gpu = use_gpu
        self.confidence_level = 0.5
        self.objectlist = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.objectlist), 3))
        self.load_model = cv2.dnn.readNetFromCaffe('modelfiles/cnnmodule.prototxt', 'modelfiles/Mobilenetssd.model')
        
        self.load_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.load_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect_objects(self):
        """
        It detect objects using the deep learning model
        shows the video feed with object detections and gives audio output for each frame of the video.
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
                self.load_model.setInput(image_feature)
                detect = self.load_model.forward()

                for i in np.arange(0, detect.shape[2]):
                    confidence = detect[0, 0, i, 2]
                    if confidence > self.confidence_level:
                        class_of_object = int(detect[0, 0, i, 1])
                        bounding_box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (topleft_x, topleft_y, downright_x, downright_y) = bounding_box.astype("int")

                        xavg = (topleft_x + downright_x) / 2
                        yavg = (topleft_y + downright_y) / 2
                        direction = get_direction(xavg, yavg)

                        bounding_box_area = (downright_x - topleft_x) * (downright_y - topleft_y) * 0.0002645833 * 0.0002645833
                        object_size = calculate_object_size(self.objectlist[class_of_object])

                        distance = calculate_distance(downright_x - topleft_x, downright_y - topleft_y, object_size, bounding_box_area, xavg)

                        label = f"{self.objectlist[class_of_object]}: {confidence * 100:.2f}%  {direction}  {distance:.2f}m"
                        self.print_detection_text(self.objectlist[class_of_object], confidence, direction, distance)
                        audio_output(label)

                        cv2.rectangle(frame, (topleft_x, topleft_y), (downright_x, downright_y), self.COLORS[class_of_object], 2)
                        y = topleft_y - 15 if topleft_y - 15 > 15 else topleft_y + 15
                        cv2.putText(frame, label, (topleft_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[class_of_object], 2)

                frame = imutils.resize(frame, width=400)
                cv2.imshow('Live detection', frame)

                if cv2.waitKey(1) == 5:
                    break

                fps.update()

        fps.stop()
    
    def print_detection_text(self, object_class, confidence, direction, distance):
        """
        Print the detection information to the console.

        Args:
            object_class (str): Class label of the detected object.
            confidence (float): Confidence level of the detected object.
            direction (str): Direction of the detected object.
            distance (float): Distance of the detected object from the camera.
        """
        text = f"THE object name is {object_class}: accuracy found is {confidence * 100:.2f}% " \
               f"{direction} distance is {distance:.2f}m"
        print(text)

if __name__ == "__main__":
    detector = ObjectDetection()
    detector.detect_objects()
