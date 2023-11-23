class DistanceEstimation():
    def __init__(self) -> None:
        self.car_width = 1.76784
        self.focal_length = 543.45

    def calc_distance(self, w):
        distance = (self.car_width * self.focal_length) / w
        return distance  # in meters
