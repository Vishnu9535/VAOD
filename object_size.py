
def calculate_object_size( object_class):
        """
        Calculate the size of the detected object based on the obbject list.
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