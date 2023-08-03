import math

focal_length = 0.026  

def calculate_distance(x_dist, y_dist, object_size, bounding_box_area, xavg):
        """
        Calculate the distance of the detected object from the camera.
        Distance of the detected object from the camera.
        """
        distance_p = (object_size * focal_length) / math.sqrt(bounding_box_area)
        width = xavg * 0.0002645833
        distance = math.sqrt(((distance_p * distance_p) + (width * width)))
        return distance