import cv2
import imutils
import math
import numpy as np
from imutils.video import FPS
import pyttsx3


class ObjectDetection:
    def __init__(self):
        """
        Initializing the class.
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
        self.load_model = cv2.dnn.readNetFromCaffe(
            'modelfiles/cnnmodule.prototxt', 'modelfiles/Mobilenetssd.model')
        self.focal_length = 0.026
        self.detect_objects()

    def detect_objects(self):
        """
        Detects objects using the deep learning model.
        """
        fps = FPS().start()
        status = True

        video_obj = cv2.VideoCapture('videos/test1.mp4')

        while status:
            status, frame = video_obj.read()
            if status:
                frame = imutils.resize(frame, width=400)
                (h, w) = frame.shape[:2]

                image_feature = cv2.dnn.blobFromImage(
                    frame, 0.007843, (300, 300), 127.5)
                self.load_model.setInput(image_feature)
                detect = self.load_model.forward()

                for i in np.arange(0, detect.shape[2]):
                    confidence = detect[0, 0, i, 2]
                    if confidence > self.confidence_level:
                        class_of_object = int(detect[0, 0, i, 1])
                        bounding_box = detect[0, 0, i,
                                              3:7] * np.array([w, h, w, h])
                        (topleft_x, topleft_y, downright_x,
                         downright_y) = bounding_box.astype("int")

                        xavg = (topleft_x + downright_x) / 2
                        yavg = (topleft_y + downright_y) / 2

                        bounding_box_area = (
                            downright_x - topleft_x) * (downright_y - topleft_y) * 0.0002645833 * 0.0002645833
                        object_size = self.calculate_object_size(
                            self.objectlist[class_of_object])
                        direction = self.get_direction(xavg, yavg)

                        distance = self.calculate_distance(
                            object_size, bounding_box_area, xavg)

                        label = f"{self.objectlist[class_of_object]}: {confidence * 100:.2f}%  {direction}  {distance:.2f}m"
                        self.print_output_text(
                            self.objectlist[class_of_object], confidence, direction, distance)
                        self.audio_output(label)
                        cv2.rectangle(frame, (topleft_x, topleft_y), (downright_x,
                                      downright_y), self.COLORS[class_of_object], 2)
                        y = topleft_y - 15 if topleft_y - 15 > 15 else topleft_y + 15
                        cv2.putText(frame, label, (topleft_x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[class_of_object], 2)

                frame = imutils.resize(frame, width=400)
                cv2.imshow('Live detection', frame)

                if cv2.waitKey(1) == 5:
                    break

                fps.update()

        fps.stop()

    def calculate_object_size(self, object_class):
        """
        Calculates the size of the detected object based on the object detected.
        """
        if object_class == "bus":
            return 36
        elif object_class == "car":
            return 10
        elif object_class == "person":
            return 1.8 * 0.4
        elif object_class == "motorbike":
            return 2.16 * 1.27
        elif object_class == "sofa":
            return 1.6002 * 0.6858
        else:
            return 5 * 5

    def calculate_distance(self, object_size, bounding_box_area, xavg):
        """
        calculates the distance of the detected object from the camera using the principle of parllax
        """
        distance_p = (object_size * self.focal_length) / \
            math.sqrt(bounding_box_area)
        width = xavg * 0.0002645833
        distance = math.sqrt(((distance_p * distance_p) + (width * width)))
        return distance

    def get_direction(self, xavg, yavg):
        """
            gives the direction of the detected object relative to the camera.
        """
        direction = "unknown"
        if xavg > 200 and yavg > 180:
            direction = "topright"
        elif xavg > 200 and yavg < 180:
            direction = "downright"
        elif xavg < 200 and yavg > 180:
            direction = "topleft"
        elif xavg < 200 and yavg < 180:
            direction = "downleft"
        return direction

    def audio_output(self, text):
        """gives the audio ouput """
        # eng = pyttsx3.init()
        # rate = eng.getProperty('rate')
        # eng.setProperty('rate', rate - 15)
        # eng.say(text)
        # eng.runAndWait()
        ...

    def print_output_text(self, object_class, confidence, direction, distance):
        """
        Prints the final output.
        """
        text = f"THE object name is {object_class}: accuracy found is {confidence * 100:.2f}% " \
               f"{direction} distance is {distance:.2f}m"
        print(text)


if __name__ == "__main__":
    detector = ObjectDetection()
