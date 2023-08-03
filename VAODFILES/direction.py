def get_direction(xavg, yavg):
        """
        Calculatees the direction of the detected object relative to the camera
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
    